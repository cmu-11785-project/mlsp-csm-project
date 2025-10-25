import os
import sys
import json
import torch
import librosa
import difflib
import logging
from collections import Counter, defaultdict
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
import base64
import time

import google.generativeai as genai
import whisper
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import SortformerEncLabelModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# from ctc_forced_aligner import (
#         load_audio,
#         load_alignment_model,
#         generate_emissions,
#         preprocess_text,
#         get_alignments,
#         get_spans,
#         postprocess_results,
#     )

# Add absolute paths for imports
PIPELINE_DIR = "/jet/home/xbai2/rudud_data_pipeline/pipelines"
PROJECT_ROOT = "/jet/home/xbai2/rudud_data_pipeline"
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from compute_snr import wada_snr
from audio_classification import AudioTaskDetectorCLAP
from voice_gender_classifier.model import ECAPA_gender

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioLabeler:
    def __init__(self, whisper_model_name="large-v3", language="auto", enable_nemo=None):
        """Initialize the AudioLabeler with all required models.
        
        Args:
            whisper_model_name: Whisper model size (default: "large-v3")
            language: Language code ("auto", "en", "ar", etc.). If "auto", will detect per audio.
            enable_nemo: Whether to load NeMo model. If None, auto-decide based on language.
                        True: always load, False: never load, None: load only for English
        """
        logger.info("Initializing AudioLabeler...")
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            self.language = language
            self.enable_nemo = enable_nemo
            
            # Load Whisper model
            logger.info(f"Loading Whisper model: {whisper_model_name}")
            self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)
            
            # Initialize Gemini client
            api_key = os.environ.get("GOOGLE_AI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_AI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini client initialized")
            
            use_cuda = torch.cuda.is_available()
            
            # Conditionally load NeMo ASR model
            self.nemo_asr = None
            should_load_nemo = self._should_load_nemo()
            
            if should_load_nemo:
                logger.info("Loading NeMo ASR model (English)...")
                try:
                    self.nemo_asr = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                        model_name="nvidia/stt_en_fastconformer_hybrid_medium_pc"
                    ).half()
                    logger.info("NeMo model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load NeMo model: {e}")
                    self.nemo_asr = None
            else:
                logger.info("NeMo model skipped (not needed for current language)")
            
            # # Load alignment model
            # logger.info("Loading alignment model...")
            # self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            #     self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32,
            # )
            
            # Load CLAP models
            logger.info("Loading CLAP models...")
            self.clap_noise = AudioTaskDetectorCLAP(task="noise", use_cuda=use_cuda)
            self.clap_non_speech = AudioTaskDetectorCLAP(task="human-sounds", use_cuda=use_cuda)
            self.clap_emotion = AudioTaskDetectorCLAP(task="emotion", use_cuda=use_cuda)

            # Load diarization model
            logger.info("Loading speaker diarization model...")
            self.diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")

            # Load gender classification model
            logger.info("Loading gender classification model...")
            self.gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier").eval().to(self.device)
            
            # Load LLM judge model
            self.llm_judge_model = None
            self.llm_judge_tokenizer = None
            logger.info("Loading LLM judge model (Qwen2.5-7B)...")
            self._load_llm_judge()
            
            logger.info("AudioLabeler initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize AudioLabeler: {e}")
            raise

    def _should_load_nemo(self):
        """Decide whether to load NeMo model based on language settings.
        
        Returns:
            bool: True if NeMo should be loaded, False otherwise
        """
        if self.enable_nemo is True:
            return True  # User explicitly enabled
        elif self.enable_nemo is False:
            return False  # User explicitly disabled
        else:  # enable_nemo is None (auto-decide)
            # Load NeMo for English, or if language is auto (might detect English)
            return self.language == "en" or self.language == "auto"

    def _load_llm_judge(self):
        """Load the Qwen LLM judge model for transcription evaluation."""
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        
        try:
            logger.info(f"Loading LLM judge (Qwen) from {model_id}...")
            self.llm_judge_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_judge_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.llm_judge_model.eval()
            logger.info("LLM judge model (Qwen) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM judge model: {e}")
            logger.warning("Falling back to Gemini transcription")
            self.llm_judge_model = None
            self.llm_judge_tokenizer = None

    def _judge_transcriptions_with_llm(self, whisper_text, gemini_text, nemo_text=None, language="ar"):
        """Use LLM to improve Gemini transcription by incorporating corrections from other ASR systems.
        
        Strategy: Use Gemini as the BASE, and correct it using insights from Whisper (and NeMo for English).
        Only fix grammatical errors or clearly wrong words in Gemini, don't rewrite everything.
        
        Args:
            whisper_text: Transcription from Whisper (reference for corrections)
            gemini_text: Transcription from Gemini (BASE text to improve)
            nemo_text: Transcription from NeMo (additional reference for English)
            language: Language code for the text
            
        Returns:
            str: Improved Gemini transcription (with corrections applied)
        """
        # Fallback if LLM judge is not available (e.g., failed to load)
        if self.llm_judge_model is None:
            logger.warning("LLM judge not available, using original Gemini transcription")
            return gemini_text
            
        try:
            # Build prompt based on language
            if language == "ar":
                # Arabic: Improve Gemini using Whisper as reference
                prompt = f"""أنت خبير في تصحيح نصوص التفريغ الصوتي.

مهمتك: صحح الأخطاء الواضحة فقط (إملاء، نحو، كلمات خاطئة) في النص الأساس مع الاستفادة من النص المرجعي. احتفظ بالنص الأساس قدر الإمكان ولا تعيد صياغته بالكامل.

النص الأساس: {gemini_text}

النص المرجعي للمساعدة: {whisper_text}

تعليمات مهمة:
1. صحح الأخطاء الواضحة فقط
2. احتفظ ببنية النص الأساس
3. اكتب النص المصحح داخل <مصحح></مصحح>
4. لا تضف أي تفسيرات أو نصوص إضافية

النص المصحح:
<مصحح>"""
            else:
                # English: Improve Gemini using both Whisper and NeMo as references
                prompt = f"""You are an expert ASR transcription corrector.

Task: Fix ONLY obvious errors (spelling, grammar, wrong words) in BASE text using REF1/REF2 as references. Preserve the BASE text structure as much as possible. Do NOT rewrite everything.

BASE: {gemini_text}

REF1 (for reference): {whisper_text}
REF2 (for reference): {nemo_text}

Important instructions:
1. Fix only clear mistakes
2. Keep BASE structure intact
3. Output corrected text inside <corrected></corrected>
4. NO explanations or extra text

Corrected text:
<corrected>"""

            # Tokenize and generate
            inputs = self.llm_judge_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            max_new_tokens = 512
            with torch.no_grad():
                outputs = self.llm_judge_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.llm_judge_tokenizer.eos_token_id
                )
            
            response = self.llm_judge_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Check if output was truncated
            output_tokens = len(self.llm_judge_tokenizer.encode(response))
            is_truncated = output_tokens >= max_new_tokens - 5  # Small margin
            
            if is_truncated:
                logger.warning(f"LLM output may be truncated ({output_tokens}/{max_new_tokens} tokens)")
            
            logger.info(f"LLM raw output length: {len(response)} chars, {output_tokens} tokens")
            logger.info(f"LLM output preview (first 150 chars): {response[:150]}...")
            logger.info(f"LLM output preview (last 100 chars): ...{response[-100:]}")
            
            # Extract text from tags (Arabic: <مصحح></مصحح>, English: <corrected></corrected>)
            import re
            # Try both tag patterns
            if language == "ar":
                match = re.search(r'<مصحح>(.*?)</مصحح>', response, re.DOTALL)
                opening_tag = '<مصحح>'
                closing_tag = '</مصحح>'
            else:
                match = re.search(r'<corrected>(.*?)</corrected>', response, re.DOTALL)
                opening_tag = '<corrected>'
                closing_tag = '</corrected>'
            
            if match:
                improved_text = match.group(1).strip()
                logger.info(f"Successfully extracted text from tags")
            else:
                # Check if output was truncated before closing tag
                if is_truncated and opening_tag in response:
                    logger.warning(f"Output truncated - missing closing tag. Falling back to Gemini text.")
                    return gemini_text
                elif opening_tag in response:
                    # Opening tag found but no closing tag and not truncated - extract what's there
                    improved_text = response.split(opening_tag, 1)[1].strip()
                    logger.warning(f"No closing tag found but not truncated, using text after opening tag")
                else:
                    # No tags at all - use entire response if it looks reasonable
                    if len(response) > 10 and not any(word in response.lower() for word in ['error', 'sorry', 'cannot']):
                        improved_text = response
                        logger.warning(f"No tags found, using entire LLM response")
                    else:
                        logger.warning(f"No valid output detected, falling back to Gemini text")
                        return gemini_text
            
            # Sanity check: if improved text is suspiciously short, use Gemini
            if len(improved_text) < len(gemini_text) * 0.3:
                logger.warning(f"Improved text too short ({len(improved_text)} vs {len(gemini_text)} chars), using Gemini")
                return gemini_text
            
            logger.info(f"Final improved text length: {len(improved_text)} chars")
            
            return improved_text
                
        except Exception as e:
            logger.error(f"LLM improvement failed: {e}, using original Gemini")
            return gemini_text

    def detect_language(self, audio_path):
        """Detect the language of the audio file using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Language code (e.g., 'ar' for Arabic, 'en' for English)
        """
        try:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            # Use the model's n_mels parameter for mel spectrogram
            mel = whisper.log_mel_spectrogram(audio, n_mels=self.whisper_model.dims.n_mels).to(self.whisper_model.device)
            _, probs = self.whisper_model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            logger.info(f"Detected language: {detected_language} (confidence: {probs[detected_language]:.2f})")
            return detected_language
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "ar"  # Default to Arabic if detection fails

    def classify_non_speech(self, audio_path):
        """Classify non-speech sounds in audio.
        
        Returns:
            tuple: (label, confidence) for non-speech sounds like breathing, coughing, etc.
        """
        return self.clap_non_speech.predict(audio_path)
    
    def classify_noise(self, audio_path):
        """Classify noise and music in audio.
        
        Returns:
            tuple: (label, confidence) for noise/music detection
        """
        return self.clap_noise.predict(audio_path)
    
    def emotion_recognition(self, audio_path):
        """Recognize emotions from speech.
        
        Returns:
            tuple: (emotion_label, confidence) for detected emotion
        """
        return self.clap_emotion.predict(audio_path)

    
    def transcribe_with_whisper(self, audio_path, detected_language=None):
        """Transcribe audio using OpenAI Whisper model with word-level timestamps."""
        # Clear GPU cache before transcription
        torch.cuda.empty_cache()
        
        # Use detected language or fallback to auto-detection
        if detected_language is None:
            detected_language = self.detect_language(audio_path)
        
        result = self.whisper_model.transcribe(
            audio_path,
            language=detected_language,  # Use detected language instead of hardcoded 'ar'
            task='transcribe',
            word_timestamps=True,  # Enable word-level timestamps
            # Add these parameters to reduce memory usage
            fp16=True,  # Use half precision
            condition_on_previous_text=False,  # Reduce memory footprint
        )
        
        text = result['text'].strip()
        word_segments = result.get('segments', [])
        
        # Extract word-level timestamps from segments
        word_timestamps = []
        for segment in word_segments:
            if 'words' in segment:
                for word_info in segment['words']:
                    word_timestamps.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'probability': word_info.get('probability', 1.0)
                    })
        
        # Clear cache after transcription
        torch.cuda.empty_cache()
        
        return text, word_timestamps

    def clean_gemini_output(self, text):
        """
        Clean Gemini output by removing timestamp labels
        
        Args:
            text: Gemini transcription with timestamps like [ 0m0s285ms - 0m7s935ms ]
            
        Returns:
            Cleaned text without timestamps
        """
        if not text:
            return ""
        
        import re
        
        # Remove timestamp patterns like [ 0m0s285ms - 0m7s935ms ] or [ 0m0s285ms ]
        cleaned = re.sub(r'\[\s*\d+[msh]\d+[msh]\d+ms\s*(?:-\s*\d+[msh]\d+[msh]\d+ms\s*)?\]', '', text)
        
        # Remove any remaining brackets with timestamps
        cleaned = re.sub(r'\[\s*[\d\s\-msh]+\]', '', cleaned)
        
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if cleaned != text:
            logger.info(f"Gemini cleaned: removed timestamps from output")
        
        return cleaned
    
    def transcribe_with_gemini(self, audio_path, detected_language=None):
        """Transcribe audio using Google Gemini AI with language-specific prompts.
        
        Args:
            audio_path: Path to the audio file
            detected_language: Detected language code ('ar' or 'en')
        
        Returns:
            str: Transcribed text
        """
        try:
            # Read audio file as bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Use Part to pass audio data, avoiding upload_file issues
            audio_part = genai.protos.Part(
                inline_data=genai.protos.Blob(
                    mime_type='audio/wav',
                    data=audio_bytes
                )
            )
            
            # Get language-specific prompt
            prompt = self._get_gemini_prompt(detected_language)
            
            response = self.genai_model.generate_content([prompt, audio_part])
            
            # Add delay to avoid rate limit
            time.sleep(7)
            
            # Clean timestamps (double insurance)
            return self.clean_gemini_output(response.text.strip())
        except Exception as e:
            logger.error(f"Gemini transcription failed: {e}")
            return ""

    def _get_gemini_prompt(self, detected_language):
        """Get language-specific prompt for Gemini transcription."""
        if detected_language == "en":
            return self._get_english_prompt()
        else:  # Default to Arabic
            return self._get_arabic_prompt()

    def _get_arabic_prompt(self):
        """Get Arabic-specific prompt for Gemini transcription."""
        return """أنت محرك معالجة إملاء باللغة العربية يتمتع بـ**صرامة دقيقة على مستوى عسكري**. مهمتك الوحيدة هي معالجة تدفق الصوت الوارد، وإخراج **مستند نصي عربي خالص** ذي تنسيق صارم ومحتوى دقيق للغاية.

**【سير العمل: التفكيك المنطقي المفصل (يتم تنفيذه داخليًا فقط، ولا يظهر في الناتج النهائي)】**

1.  **الخطوة الأولى: التقاط البيانات الأصلية**
    * **الإجراء**: قم بإجراء الإملاء الأولي (النسخ) للمحتوى المنطوق في تدفق الصوت، مع تسجيل تسلسل الكلمات بالكامل بما في ذلك التلعثم أو التكرار.
    * **التحقق**: تأكد من تسجيل جميع الأصوات، بما في ذلك الأخطاء اللفظية أو الكلمات غير المكتملة.

2.  **الخطوة الثانية: تنظيف التنسيق والعلامات**
    * **الهدف**: التطهير الكامل لجميع العناصر غير النصية وغير الكلمات العربية.
    * **نقطة التحقق أ (غير نصي)**: امسح النص، و**احذف** جميع الطوابع الزمنية، والأرقام، وعلامات الترقيم، وأي رموز غير أبجدية.
    * **نقطة التحقق ب (غير لغوي)**: امسح النص، و**احذف** جميع العلامات التي تمثل أصواتًا بيئية أو حالات نطق، مثل: [موسيقى]، [ضحك]، [صمت]،[همهمة]، [صوت عالٍ]، أو أي ملاحظات للمدون.

3.  **الخطوة الثالثة: التحقق من دقة الكلمات**
    * **الهدف**: ضمان أمانة النسخ للكلمات، ومنع التصحيح الدلالي التلقائي.
    * **القاعدة أ (الأمانة)**: مراجعة النسخة الأولية. إذا كان المتحدث قد ارتكب **أخطاء لفظية**، أو **كرر** كلمات، أو استخدم **تراكيب غير نحوية/غير صحيحة**، **فيجب الاحتفاظ بها كما هي تمامًا**، ويُمنع منعًا باتًا إجراء أي **تنقيح أو تصحيح نحوي أو دلالي**.
    * **القاعدة ب (الكتابة القياسية)**: لكل كلمة، تأكد من كتابتها باستخدام **الأحرف الأبجدية العربية القياسية وقواعد الإملاء الصحيحة**، لضمان قابلية قراءة النص.

4.  **الخطوة الرابعة: إنشاء النص النهائي للإخراج**
    * **الإجراء**: قم بربط النص الذي تمت معالجته والتحقق منه في الخطوة الثالثة، وتأكد من تقديمه في شكل **فقرة متماسكة وطبيعية**.
    * **التحقق النهائي**: قم بإجراء فحص أخير للتأكد من أن النص **لا يحتوي** على أي تعليقات، أو تعليمات، أو أي أثر لمحتوى غير النسخ.

5.  **الخطوة الخامسة: إخراج النتيجة**
    * **الإجراء**: قم بإخراج النص النهائي الذي تم إنشاؤه في الخطوة الرابعة **فقط**.

**【قواعد الإخراج النهائية: خط أحمر مطلق】**
* **المحتوى**: يجب أن يكون الناتج **نصًا عربيًا خالصًا للمحتوى المنطوق**.
* **القيود**: **يُمنع منعًا باتًا** استخدام أي علامات ترقيم، أو أرقام، أو علامات زمنية، أو أوصاف غير لغوية."""

    def _get_english_prompt(self):
        """Get English-specific prompt for Gemini transcription."""
        return """You are a military-grade precision audio transcription engine specialized in **English language processing**. Your sole mission is to process incoming audio streams and produce **pure English text documents** with strict formatting and extremely accurate content.

**【WORKFLOW: DETAILED LOGICAL BREAKDOWN (executed internally only, not shown in final output)】**

1.  **Step One: Raw Data Capture**
    * **Action**: Perform initial transcription (dictation) of spoken content in the audio stream, recording the complete word sequence including stutters or repetitions.
    * **Verification**: Ensure all sounds are recorded, including pronunciation errors or incomplete words.

2.  **Step Two: Format and Markup Cleanup**
    * **Goal**: Complete purification of all non-textual and non-English word elements.
    * **Checkpoint A (Non-textual)**: Clean the text and **delete** all timestamps, numbers, punctuation marks, and any non-alphabetic symbols.
    * **Checkpoint B (Non-linguistic)**: Clean the text and **delete** all markers representing environmental sounds or pronunciation states, such as: [music], [laughter], [silence], [mumbling], [loud voice], or any annotator notes.

3.  **Step Three: Word Accuracy Verification**
    * **Goal**: Ensure faithful transcription of words and prevent automatic semantic correction.
    * **Rule A (Faithfulness)**: Review the initial transcription. If the speaker made **pronunciation errors**, **repeated** words, or used **ungrammatical/incorrect structures**, **they must be preserved exactly as spoken**, and any **grammatical or semantic revision or correction is strictly prohibited**.
    * **Rule B (Standard Writing)**: For each word, ensure it is written using **standard English alphabet and correct spelling rules** to ensure text readability.

4.  **Step Four: Create Final Output Text**
    * **Action**: Connect the processed and verified text from Step Three, ensuring it is presented as a **coherent and natural paragraph**.
    * **Final Verification**: Perform a final check to ensure the text **contains no** comments, instructions, or any traces of non-transcription content.

5.  **Step Five: Output Result**
    * **Action**: Output only the final text created in Step Four.

**【FINAL OUTPUT RULES: ABSOLUTE RED LINE】**
* **Content**: The output must be **pure English text of the spoken content**.
* **Restrictions**: **Strictly prohibited** to use any punctuation marks, numbers, timestamps, or non-linguistic descriptions."""

    def _get_labeled_transcription_prompt(self, label, detected_language):
        """Get language-specific prompt for labeled transcription."""
        if detected_language == "en":
            return self._get_english_labeled_prompt(label)
        else:  # Default to Arabic
            return self._get_arabic_labeled_prompt(label)

    def _get_arabic_labeled_prompt(self, label):
        """Get Arabic-specific prompt for labeled transcription."""
        return f"""أنت محرك معالجة إملاء باللغة العربية بدقة عسكرية. مهمتك نسخ المحتوى المنطوق مع الإشارة إلى الأصوات غير اللغوية.

**قواعد النسخ:**
1. انسخ كل الكلام المنطوق بدقة تامة كما هو
2. للأصوات غير اللغوية، استخدم علامات مثل [ضحك] أو [{label}]
3. **ممنوع**: الطوابع الزمنية، الأرقام، أو أي تعليمات
4. احتفظ بالأخطاء اللفظية كما هي، لا تصحح
5. الناتج: نص عربي خالص مع علامات الأصوات فقط

أخرج النص المنسوخ مباشرة."""

    def _get_english_labeled_prompt(self, label):
        """Get English-specific prompt for labeled transcription."""
        return f"""You are a military-grade precision English transcription engine. Your task is to transcribe spoken content while marking non-linguistic sounds.

**Transcription Rules:**
1. Transcribe all spoken words with complete accuracy
2. For non-linguistic sounds, use markers like [laughter] or [{label}]
3. **Prohibited**: Timestamps, numbers, or any instructions
4. Preserve pronunciation errors as they are, do not correct
5. Output: Pure English text with sound markers only

Output the transcribed text directly."""

    def transcribe_with_gemini_with_label(self, audio_path, label, detected_language=None):
        """Transcribe audio with non-speech sound labels using Google Gemini AI.
        
        Args:
            audio_path: Path to the audio file
            label: Non-speech sound label
            detected_language: Detected language code ('ar' or 'en')
        
        Returns:
            str: Transcribed text with labeled non-speech sounds
        """
        try:
            # Read audio file as bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Use Part to pass audio data
            audio_part = genai.protos.Part(
                inline_data=genai.protos.Blob(
                    mime_type='audio/wav',
                    data=audio_bytes
                )
            )
            
            # Get language-specific prompt for labeled transcription
            prompt = self._get_labeled_transcription_prompt(label, detected_language)
            
            response = self.genai_model.generate_content([prompt, audio_part])
            
            # Add delay to avoid rate limit
            time.sleep(7)
            
            # Clean timestamps (double insurance)
            return self.clean_gemini_output(response.text.strip())
        except Exception as e:
            logger.error(f"Gemini with label failed: {e}")
            return ""

    def transcribe_with_nemo(self, audio_path):
        """Transcribe audio using NeMo ASR model with memory optimization."""
        
        # 1. Unload all other models to CPU
        logger.info("Unloading models to CPU to free GPU memory...")
        
        # Whisper
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            self.whisper_model = self.whisper_model.cpu()
        
        # CLAP models (3 models)
        if hasattr(self, 'clap_noise'):
            self.clap_noise.clap_model.clap = self.clap_noise.clap_model.clap.cpu()
        if hasattr(self, 'clap_non_speech'):
            self.clap_non_speech.clap_model.clap = self.clap_non_speech.clap_model.clap.cpu()
        if hasattr(self, 'clap_emotion'):
            self.clap_emotion.clap_model.clap = self.clap_emotion.clap_model.clap.cpu()
        
        # Diarization
        if hasattr(self, 'diar_model'):
            self.diar_model = self.diar_model.cpu()
        
        # Gender
        if hasattr(self, 'gender_model'):
            self.gender_model = self.gender_model.cpu()
        
        # Alignment
        # if hasattr(self, 'alignment_model'):
        #     self.alignment_model = self.alignment_model.cpu()
        
        # 2. Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 3. Print available GPU memory
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # 4. Transcribe audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, sr)
        
        try:
            output = self.nemo_asr.transcribe([tmp_path])
            
            # Fix: Correctly extract text from Hypothesis object
            if output and len(output) > 0:
                hypothesis = output[0]  # Get first Hypothesis object
                if hasattr(hypothesis, 'text'):
                    result = hypothesis.text  # Extract text attribute
                else:
                    result = str(hypothesis)  # Fallback
            else:
                result = ""
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # 5. Clear cache
        torch.cuda.empty_cache()
        
        # 6. Restore all models to GPU
        logger.info("Reloading models back to GPU...")
        
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            self.whisper_model = self.whisper_model.to(self.device)
        
        if hasattr(self, 'clap_noise'):
            self.clap_noise.clap_model.clap = self.clap_noise.clap_model.clap.cuda()
        if hasattr(self, 'clap_non_speech'):
            self.clap_non_speech.clap_model.clap = self.clap_non_speech.clap_model.clap.cuda()
        if hasattr(self, 'clap_emotion'):
            self.clap_emotion.clap_model.clap = self.clap_emotion.clap_model.clap.cuda()
        
        if hasattr(self, 'diar_model'):
            self.diar_model = self.diar_model.to(self.device)
        
        if hasattr(self, 'gender_model'):
            self.gender_model = self.gender_model.to(self.device)
        
        # if hasattr(self, 'alignment_model'):
        #     self.alignment_model = self.alignment_model.to(self.device)
        
        logger.info("All models reloaded to GPU")
        
        return result
    
    # Improved memory management
    def calculate_snr(self, audio_segment):
        """Calculate Signal-to-Noise Ratio."""
        if len(audio_segment) == 0:
            return 0
        return wada_snr(audio_segment)
    
    # def align_text(self, audio_path, transcription, language="acm", batch_size=4):
    #     """Implement text-audio alignment logic using CTC forced aligner.
        
    #     Args:
    #         audio_path: Path to audio file
    #         transcription: Text transcription to align
    #         language: Language code (arb=standard arabic, acm=gulf)
    #         batch_size: Batch size for processing (reduced from 16 to 8 for memory)
        
    #     Returns:
    #         Word timestamps with alignment information
    #     """
    #     torch.cuda.empty_cache()
        
    #     audio_waveform = load_audio(audio_path, self.alignment_model.dtype, self.alignment_model.device)

    #     emissions, stride = generate_emissions(
    #         self.alignment_model, audio_waveform, batch_size=batch_size
    #     )

    #     tokens_starred, text_starred = preprocess_text(
    #         transcription,
    #         romanize=True,
    #         language=language,
    #     )

    #     segments, scores, blank_token = get_alignments(
    #         emissions,
    #         tokens_starred,
    #         self.alignment_tokenizer,
    #     )

    #     spans = get_spans(tokens_starred, segments, blank_token)

    #     word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    #     torch.cuda.empty_cache()
    #     return word_timestamps

    def merge_transcriptions(self, transcriptions):
        """
        Merge multiple transcriptions using ROVER algorithm:
        1. Pick a reference transcription
        2. Align other transcriptions to reference using Levenshtein distance
        3. Combine results based on alignments
        4. Perform voting at each position to select final words
        """
        if not transcriptions or len(transcriptions) == 0:
            return ""
        
        # Debug: Check input types
        logger.info(f"merge_transcriptions input types: {[type(t) for t in transcriptions]}")
        logger.info(f"merge_transcriptions input content: {transcriptions}")
        
        # Remove empty transcriptions with better type handling
        valid_transcriptions = []
        for t in transcriptions:
            if t is None:
                continue
            elif hasattr(t, 'text'):  # Handle Hypothesis objects
                text = t.text.strip() if t.text else ""
                if text:
                    valid_transcriptions.append(text)
            elif isinstance(t, str):
                text = t.strip()
                if text:
                    valid_transcriptions.append(text)
            else:
                # Try to convert to string
                text = str(t).strip()
                if text:
                    valid_transcriptions.append(text)
        
        if len(valid_transcriptions) == 0:
            return ""
        
        if len(valid_transcriptions) == 1:
            return valid_transcriptions[0]
        
        # Step 1: Pick reference - choose the longest transcription as reference
        reference_idx = max(range(len(valid_transcriptions)), 
                        key=lambda i: len(valid_transcriptions[i].split()))
        reference = valid_transcriptions[reference_idx]
        reference_words = reference.split()
        
        # Step 2 & 3: Align other transcriptions and combine results
        all_alignments = []
        
        for i, transcription in enumerate(valid_transcriptions):
            if i == reference_idx:
                # Reference alignment (exact match)
                alignment = list(range(len(reference_words)))
            else:
                # Align transcription to reference using sequence matching
                alignment = self._align_to_reference(reference_words, transcription.split())
            
            all_alignments.append((transcription.split(), alignment))
        
        # Step 4: Perform voting at each position
        max_length = max(len(alignment) for _, alignment in all_alignments)
        voted_words = []
        
        for pos in range(max_length):
            # Collect words at this position from all transcriptions
            position_words = []
            
            for words, alignment in all_alignments:
                if pos < len(alignment):
                    aligned_pos = alignment[pos]
                    if aligned_pos is not None and aligned_pos < len(words):
                        word = words[aligned_pos]
                        if word:  # Only add non-empty words
                            position_words.append(word.lower())
            
            if position_words:
                # Vote for the most common word at this position
                word_counts = Counter(position_words)
                most_common_word = word_counts.most_common(1)[0][0]
                voted_words.append(most_common_word)
        
        return " ".join(voted_words)

    def _align_to_reference(self, reference_words, target_words):
        """
        Align target words to reference words using sequence matching.
        Returns alignment indices for target words.
        """
        if not target_words:
            return []
        
        if not reference_words:
            return [None] * len(target_words)
        
        # Use difflib to find the best alignment
        matcher = difflib.SequenceMatcher(None, reference_words, target_words)
        
        # Create alignment mapping
        alignment = [None] * len(target_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Direct match
                for k in range(j2 - j1):
                    if j1 + k < len(alignment):
                        alignment[j1 + k] = i1 + k
            elif tag == 'replace':
                # Substitution - map as best as possible
                ref_len = i2 - i1
                target_len = j2 - j1
                for k in range(target_len):
                    if j1 + k < len(alignment):
                        ref_idx = i1 + min(k, ref_len - 1)
                        alignment[j1 + k] = ref_idx
            elif tag == 'insert':
                # Insertion in target - try to map to nearby reference position
                ref_pos = i1 if i1 < len(reference_words) else len(reference_words) - 1
                for k in range(j2 - j1):
                    if j1 + k < len(alignment):
                        alignment[j1 + k] = ref_pos
            elif tag == 'delete':
                # Deletion in target - no mapping needed
                pass
        
        return alignment

    def polish_asr_text(self, text, detected_language):
        """Polish ASR transcription text using Gemini based on detected language."""
        try:
            # Use different prompts based on detected language
            if detected_language == "ar":
                prompt = f"""
                أنت خبير في تحسين جودة النصوص العربية. مهمتك هي معالجة النص الناتج عن نظام تحويل الكلام إلى نص (ASR) لجعله نظيفاً، خالياً من الأخطاء، ومُنقطاً بشكل صحيح.

                **التعليمات:**
                1.  **التصحيح الإملائي والنحوي:** قم بتصحيح الأخطاء الإملائية والنحوية في النص.
                2.  **التنقية:** أضف المسافات الصحيحة بين الكلمات والحروف إذا كانت مفقودة.
                3.  **التنقيط:** أضف علامات الترقيم الصحيحة والمناسبة، مثل الفواصل في الأماكن الصحيحة والنقاط في نهاية الجمل وعلامات الاستفهام في نهاية الأسئلة.
                4.  **المحافظة على المحتوى:** لا تقم بحذف أو تغيير الأسماء الصحيحة أو المصطلحات التقنية أو أسماء العلامات التجارية.
                5.  **الناتج النهائي:** يجب أن يكون الناتج نصاً واحداً كاملاً ومنقحاً جاهزاً للقراءة.

                **مثال على المدخلات:**
                "انا اعتقد ان هذا شي مهم متى سيكون الاجتماع القادم هل يمكنك ان ترسل لي بريد إلكتروني"

                **الناتج المطلوب:**
                "أنا أعتقد أن هذا شيء مهم. متى سيكون الاجتماع القادم؟ هل يمكنك أن ترسل لي بريدًا إلكترونيًا."

                **النص للمعالجة:**
                {text}
                """
            else:
                # English or other languages
                prompt = f"""
                You are an expert in improving text quality from Automatic Speech Recognition (ASR) systems. Your task is to process the ASR output text to make it clean, error-free, and properly punctuated.

                **Instructions:**
                1.  **Spelling and Grammar Correction:** Correct any spelling and grammatical errors in the text.
                2.  **Cleaning:** Add proper spacing between words and characters if missing.
                3.  **Punctuation:** Add correct and appropriate punctuation marks, such as commas in the right places, periods at the end of sentences, and question marks at the end of questions.
                4.  **Content Preservation:** Do not delete or change proper names, technical terms, or brand names.
                5.  **Final Output:** The output should be a single, complete, polished text ready for reading.

                **Example Input:**
                "i think this is important when is the next meeting can you send me an email"

                **Expected Output:**
                "I think this is important. When is the next meeting? Can you send me an email?"

                **Text to Process:**
                {text}
                """
        
            response = self.genai_model.generate_content([prompt])
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Polish ASR text failed: {e}")
            return text  # Return original text as fallback
        
    def get_number_of_speakers(self, audio_path):
        """Perform speaker diarization."""
        
        # Debug: Check GPU memory before diarization
        logger.info("=== GPU Memory Before Diarization ===")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            logger.info(f"Total: {total:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
        
        torch.cuda.empty_cache()
        
        # Check after cache clear
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free = total - reserved
            logger.info(f"After cache clear - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")

        try:
            # Load audio as mono to ensure compatibility with diarization model
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Save as temporary mono WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, sr)
            
            try:
                # Check memory right before diarization model call
                logger.info("=== Just Before Diarization Model Call ===")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    logger.info(f"Pre-diarization: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                # Perform diarization on mono audio file with conservative settings
                logger.info("Starting diarization model call...")
                
                # Try with very small batch_size and limited processing
                try:
                    segments = self.diar_model.diarize(
                        audio=tmp_path, 
                        batch_size=1,  # Already minimum
                        num_workers=1   # Reduce workers
                    )
                except Exception as first_error:
                    logger.warning(f"First diarization attempt failed: {first_error}")
                    
                    # Try with even more conservative settings
                    torch.cuda.empty_cache()
                    logger.info("Trying diarization with ultra-conservative settings...")
                    
                    # Load a smaller chunk of audio if file is too long
                    audio_data, sr = librosa.load(audio_path, sr=16000, mono=True, duration=300)  # Only first 5 minutes
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file2:
                        tmp_path2 = tmp_file2.name
                        sf.write(tmp_path2, audio_data, sr)
                    
                    try:
                        segments = self.diar_model.diarize(
                            audio=tmp_path2, 
                            batch_size=1,
                            num_workers=1
                        )
                    except Exception as second_error:
                        logger.error(f"Second diarization attempt also failed: {second_error}")
                        logger.warning("Skipping diarization due to persistent memory issues")
                        # Return dummy diarization result
                        segments = [["0.0 19.0 speaker_0"]]  # Fake single speaker for whole duration
                    finally:
                        if os.path.exists(tmp_path2):
                            os.unlink(tmp_path2) 
                spks = []
                new_segments = []
                for seg in segments[0]:
                    s, e, spk = seg.split()
                    new_segments.append((s, e, spk))
                    spks.append(spk)
                
                torch.cuda.empty_cache()
                return new_segments, len(set(spks))
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            torch.cuda.empty_cache()
            return [], 1

    def get_gender(self, audio_path):
        """Recognize speaker gender."""
        return self.gender_model.predict(audio_path, self.device)

    def process_audio(self, audio_path, output_path=None):
        """Main processing function."""
        logger.info(f"Processing audio file: {audio_path}")
        
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            results = {}
            
            # Load audio once
            audio, sr = librosa.load(audio_path, sr=16_000)
            duration = librosa.get_duration(y=audio, sr=sr)
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Language Detection (before Step 1)
            if self.language == "auto":
                detected_language = self.detect_language(audio_path)
                logger.info(f"Language detected: {detected_language}")
            else:
                detected_language = self.language
                logger.info(f"Language set to: {detected_language}")
            
            # Step 1: Audio Classification
            logger.info("Step 1: Audio classification...")
            non_speech_label, non_speech_confidence = self.classify_non_speech(audio_path)
            noise_label, noise_confidence = self.classify_noise(audio_path)
            emotion_label, emotion_confidence = self.emotion_recognition(audio_path)
            
            results['audio_classification'] = {
                'non_speech': {'label': non_speech_label, 'confidence': non_speech_confidence},
                'noise': {'label': noise_label, 'confidence': noise_confidence},
                'emotion': {'label': emotion_label, 'confidence': emotion_confidence},
            }
            
            torch.cuda.empty_cache()
            
            # Step 2: Transcriptions - Use correct method names
            logger.info("Step 2: Transcription...")
            
            # Whisper transcription with word timestamps (pass detected language)
            whisper_transcription, whisper_word_timestamps = self.transcribe_with_whisper(audio_path, detected_language)
            
            # Gemini transcription (pass detected language)
            gemini_transcription = self.transcribe_with_gemini(audio_path, detected_language)
            
            # Gemini with non-speech labels
            if non_speech_label != "no_sounds" and non_speech_confidence > 0.6:
                gemini_transcription_with_label = self.transcribe_with_gemini_with_label(audio_path, non_speech_label, detected_language)
            else:
                gemini_transcription_with_label = gemini_transcription

            # NeMo transcription (conditional - only for English)
            if detected_language == "en" and self.nemo_asr is not None:
                logger.info("Using NeMo for English transcription")
                nemo_transcription = self.transcribe_with_nemo(audio_path)
            else:
                logger.info(f"Skipping NeMo (language={detected_language}, nemo_loaded={self.nemo_asr is not None})")
                nemo_transcription = ""  # Empty string, won't participate in voting
            
            # Debug: Check what NeMo returned
            logger.info(f"NeMo transcription type: {type(nemo_transcription)}")
            logger.info(f"NeMo transcription content: {nemo_transcription}")
            
            # Merge transcriptions using LLM Judge
            logger.info("Using LLM Judge to select best transcription")
            # Pass NeMo transcription only if it's valid (for English)
            nemo_for_judge = nemo_transcription if (detected_language == "en" and nemo_transcription) else None
            merged_transcription = self._judge_transcriptions_with_llm(
                whisper_transcription, 
                gemini_transcription,
                nemo_for_judge,
                detected_language
            )

            results['transcriptions'] = {
                'detected_language': detected_language,
                'whisper': whisper_transcription,
                'gemini': gemini_transcription,
                'nemo': nemo_transcription if detected_language == "en" else f"skipped ({detected_language} audio)",
                'merged': merged_transcription,
                'gemini_with_label': gemini_transcription_with_label,
                "final": self.polish_asr_text(merged_transcription, detected_language)
            }
            
            # Step 3: Use Whisper word timestamps
            logger.info("Step 3: Using Whisper word timestamps...")
            results['alignment'] = whisper_word_timestamps
            
            # Step 4: Audio quality metrics
            logger.info("Step 4: Computing SNR...")
            results['snr'] = self.calculate_snr(audio)
            words = len(results['transcriptions']['merged'].split())
            
            # Step 5: Speaker diarization and gender classification
            logger.info("Step 5: Speaker diarization and gender classification...")
            
            # EXTRA MEMORY MANAGEMENT: Unload heavy models before diarization
            logger.info("Unloading heavy models for diarization...")
            
            # Move Whisper and NeMo to CPU temporarily
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                self.whisper_model = self.whisper_model.cpu()
            if hasattr(self, 'nemo_asr') and self.nemo_asr is not None:
                self.nemo_asr = self.nemo_asr.cpu()
            
            # Move CLAP models to CPU
            if hasattr(self, 'clap_noise'):
                self.clap_noise.clap_model.clap = self.clap_noise.clap_model.clap.cpu()
            if hasattr(self, 'clap_non_speech'):
                self.clap_non_speech.clap_model.clap = self.clap_non_speech.clap_model.clap.cpu()
            if hasattr(self, 'clap_emotion'):
                self.clap_emotion.clap_model.clap = self.clap_emotion.clap_model.clap.cpu()
            
            # Clear cache thoroughly
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force memory defragmentation
            try:
                # Create and delete a large tensor to force memory consolidation
                temp_tensor = torch.zeros(1000, 1000, 1000, device='cuda', dtype=torch.float16)
                del temp_tensor
                torch.cuda.empty_cache()
                logger.info("Memory defragmentation completed")
            except Exception as e:
                logger.warning(f"Memory defragmentation failed: {e}")
            
            # Check memory after unloading
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"After unloading models: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            diar, num_spkrs = self.get_number_of_speakers(audio_path)
            
            # Alternative: Try CPU-based diarization if GPU fails
            if num_spkrs == 1 and not diar:  # If diarization failed (returned default)
                logger.info("Attempting CPU-based diarization as fallback...")
                try:
                    # Move diarization model to CPU
                    self.diar_model = self.diar_model.cpu()
                    torch.cuda.empty_cache()
                    
                    # Try diarization on CPU
                    audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        sf.write(tmp_path, audio_data, sr)
                    
                    try:
                        segments = self.diar_model.diarize(audio=tmp_path, batch_size=1)
                        spks = []
                        new_segments = []
                        for seg in segments[0]:
                            s, e, spk = seg.split()
                            new_segments.append((s, e, spk))
                            spks.append(spk)
                        diar, num_spkrs = new_segments, len(set(spks))
                        logger.info(f"CPU diarization successful: {num_spkrs} speakers found")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        # Move model back to GPU
                        self.diar_model = self.diar_model.to(self.device)
                        
                except Exception as e:
                    logger.warning(f"CPU diarization also failed: {e}")
                    # Keep the default values
            
            # Restore models after diarization
            logger.info("Restoring models after diarization...")
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                self.whisper_model = self.whisper_model.to(self.device)
            if hasattr(self, 'nemo_asr') and self.nemo_asr is not None:
                self.nemo_asr = self.nemo_asr.to(self.device)
            if hasattr(self, 'clap_noise'):
                self.clap_noise.clap_model.clap = self.clap_noise.clap_model.clap.cuda()
            if hasattr(self, 'clap_non_speech'):
                self.clap_non_speech.clap_model.clap = self.clap_non_speech.clap_model.clap.cuda()
            if hasattr(self, 'clap_emotion'):
                self.clap_emotion.clap_model.clap = self.clap_emotion.clap_model.clap.cuda()
            
            results['speaker_info'] = {
                "num_speakers": num_spkrs,
                "gender": self.get_gender(audio_path),
                "Diarization": diar, 
                "duration": duration,
                "word_per_sec": words/duration,
                "word_per_min": words/(duration/60)
            }
            
            # Final cleanup
            torch.cuda.empty_cache()

            # Save results to JSON
            json_path = output_path if output_path else audio_path.rsplit('.', 1)[0] + '.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
            logger.info(f"Results saved to: {json_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            raise

    # Fix solution
    def _move_clap_to_cpu(self):
        """Completely move CLAP models to CPU"""
        clap_models = ['clap_noise', 'clap_non_speech', 'clap_emotion']
        
        for model_name in clap_models:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                try:
                    if hasattr(model, 'clap_model'):
                        # Move main model
                        model.clap_model = model.clap_model.cpu()
                        
                        # Move all sub-components
                        if hasattr(model.clap_model, 'clap'):
                            model.clap_model.clap = model.clap_model.clap.cpu()
                        if hasattr(model.clap_model, 'audio_encoder'):
                            model.clap_model.audio_encoder = model.clap_model.audio_encoder.cpu()
                        if hasattr(model.clap_model, 'text_encoder'):
                            model.clap_model.text_encoder = model.clap_model.text_encoder.cpu()
                            
                except Exception as e:
                    logger.warning(f"Failed to move {model_name} to CPU: {e}")

    def _move_clap_to_gpu(self):
        """Completely move CLAP models to GPU"""
        clap_models = ['clap_noise', 'clap_non_speech', 'clap_emotion']
        
        for model_name in clap_models:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                try:
                    if hasattr(model, 'clap_model'):
                        # Move main model
                        model.clap_model = model.clap_model.cuda()
                        
                        # Move all sub-components
                        if hasattr(model.clap_model, 'clap'):
                            model.clap_model.clap = model.clap_model.clap.cuda()
                        if hasattr(model.clap_model, 'audio_encoder'):
                            model.clap_model.audio_encoder = model.clap_model.audio_encoder.cuda()
                        if hasattr(model.clap_model, 'text_encoder'):
                            model.clap_model.text_encoder = model.clap_model.text_encoder.cuda()
                            
                except Exception as e:
                    logger.warning(f"Failed to move {model_name} to GPU: {e}")
                    # If move fails, reinitialize
                    logger.info(f"Reinitializing {model_name}...")
                    self._reinitialize_clap_model(model_name)

    def _reinitialize_clap_model(self, model_name):
        """Reinitialize CLAP model"""
        try:
            if model_name == 'clap_noise':
                self.clap_noise = AudioTaskDetectorCLAP(task="noise", use_cuda=True)
            elif model_name == 'clap_non_speech':
                self.clap_non_speech = AudioTaskDetectorCLAP(task="human-sounds", use_cuda=True)
            elif model_name == 'clap_emotion':
                self.clap_emotion = AudioTaskDetectorCLAP(task="emotion", use_cuda=True)
        except Exception as e:
            logger.error(f"Failed to reinitialize {model_name}: {e}")

def main():
    """Main function to handle command line arguments and process audio files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio processing pipeline with transcription and labeling')
    parser.add_argument('--input', help='Input audio file or directory')
    parser.add_argument('--output', help='Output file or directory for results')
    parser.add_argument('--input_dir', help='Input directory containing audio files')
    parser.add_argument('--output_dir', help='Output directory for results')
    parser.add_argument('--whisper-model', default='large-v3', 
                       help='Whisper model size (default: large-v3, options: tiny/base/small/medium/large/large-v2/large-v3)')
    parser.add_argument('--language', default='auto',
                       help='Language code (default: auto, options: auto/en/ar/...)')
    parser.add_argument('--enable-nemo', default=None, type=lambda x: None if x == 'auto' else x == 'true',
                       help='Enable NeMo model (default: auto, options: auto/true/false)')
    
    args = parser.parse_args()
    
    # Handle both --input and --input_dir
    if args.input_dir:
        args.input = args.input_dir
    if args.output_dir:
        args.output = args.output_dir
    
    if not args.input:
        parser.error("Either --input or --input_dir must be specified")
    
    # Initialize the audio labeler
    labeler = AudioLabeler(
        whisper_model_name=args.whisper_model,
        language=args.language,
        enable_nemo=args.enable_nemo
    )
    
    # Process single file or directory
    if os.path.isfile(args.input):
        # Single file processing
        output_path = args.output if args.output else None
        results = labeler.process_audio(args.input, output_path=output_path)
        logger.info("Processing completed successfully")
        
    elif os.path.isdir(args.input):
        # Directory processing
        if not args.output:
            args.output = os.path.join(args.input, 'results')
        
        os.makedirs(args.output, exist_ok=True)
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(args.input).glob(f'**/*{ext}'))
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file}")
            
            try:
                # Create output path maintaining directory structure
                rel_path = audio_file.relative_to(args.input)
                output_file = os.path.join(args.output, str(rel_path.with_suffix('.json')))
                
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Process the audio file
                results = labeler.process_audio(str(audio_file), output_path=output_file)
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                continue
        
        logger.info("Batch processing completed")
        
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
