"""
OCR Engine — dots.ocr model wrapper.
Handles model loading, device selection, and inference.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import time as _time



# The prompt instructs the model to parse all layout elements and extract text.
DOCUMENT_PARSE_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""


class OCREngine:
    """Wrapper around dots.ocr for document text extraction."""

    def __init__(self, model_path: str):
        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            attn_impl = "flash_attention_2"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            attn_impl = "eager"
            dtype = torch.float32
        else:
            self.device = "cpu"
            attn_impl = "eager"
            dtype = torch.bfloat16

        print(f"  Device: {self.device} | Attention: {attn_impl} | Dtype: {dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )        
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval()

        

        # Patch: newer transformers requires video_processor in Qwen2.5VL processor
        # but dots.ocr's custom processor doesn't pass it -> monkey-patch the check
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        except TypeError as e:
            if "video_processor" in str(e):
                from transformers.processing_utils import ProcessorMixin

                original_check = ProcessorMixin.check_argument_for_proper_class

                def patched_check(self_proc, argument_name, arg):
                    if argument_name == "video_processor" and arg is None:
                        return
                    return original_check(self_proc, argument_name, arg)

                ProcessorMixin.check_argument_for_proper_class = patched_check
                self.processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
                ProcessorMixin.check_argument_for_proper_class = original_check
            else:
                raise

    def extract(self, image: Image.Image) -> str:
        """Run OCR on a single page image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DOCUMENT_PARSE_PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Force ALL tensors to float32 on CPU
        inputs = {
            k: v.to(device="cpu", dtype=torch.bfloat16) if v.is_floating_point() else v.to(device="cpu")
            for k, v in inputs.items()
        }

        for k, v in inputs.items():
            if hasattr(v, 'dtype'):
                print(f"    {k}: dtype={v.dtype}, shape={v.shape}")

        print(f"        Input tokens: {inputs['input_ids'].shape[1]}")
        print(f"        Pixel values shape: {inputs['pixel_values'].shape}")
        print(f"        Generating (max 4096 tokens)...", flush=True)
        gen_start = _time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                repetition_penalty=1.2
            )
        gen_time = _time.time() - gen_start
        new_tokens = generated_ids.shape[1] - inputs['input_ids'].shape[1]
        print(f"        Generated {new_tokens} tokens in {gen_time:.1f}s ({new_tokens/gen_time:.1f} tok/s)")


        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""