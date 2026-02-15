import os
import random
import cv2
import numpy as np
from PIL import Image
from docx import Document
from docx.shared import Inches, Cm

class NormalizedHandwritingEngine:
    def __init__(self, base_path="my_letters", ink_color=(20, 24, 82)):
        self.base_path = base_path
        self.ink_color = ink_color 
        self.std_size = 50       
        self.line_height = 65    
        self.char_spacing = 2    

    def process_letter_contour(self, image_path, char_ref):
        img = cv2.imread(image_path)
        if img is None: return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        if not valid_contours: valid_contours = contours

        all_points = np.concatenate(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        letter_roi = thresh[y:y+h, x:x+w]
        
        # --- FIX 1: HEAVY PRE-PADDING ---
        # Add a thick border (10px) to the raw letter.
        # This guarantees that even if we resize or dilate later, 
        # the ink is far from the image edge.
        roi_padding = 10
        letter_roi = cv2.copyMakeBorder(
            letter_roi, 
            roi_padding, roi_padding, roi_padding, roi_padding, 
            cv2.BORDER_CONSTANT, 
            value=0
        )
        # Update dimensions after padding
        h, w = letter_roi.shape 
        
        # --- GROUPS ---
        small_punct = ['.', ',', "'", '"', '`', '-']
        tall_letters = ['f', 't', 'b', 'd', 'h', 'k', 'l']
        descenders = ['g', 'j', 'p', 'q', 'y']
        # Added 's' to short letters to fix the specific clipping you saw
        short_letters = ['a', 'c', 'e', 'i', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z']
        
        char_type = char_ref.lower() 
        
        # --- FIX 2: DRASTIC SIZE REDUCTION ---
        # Previously we used 0.90 (90%). This was too big.
        # We reduce to ~60% to allow room for alignment shifts.
        
        # The "Safe Height" is the box size minus margins
        safe_zone = self.std_size - 4 

        if char_type in small_punct:
            target_height = int(safe_zone * 0.20)
        elif char_type in tall_letters:
            target_height = int(safe_zone * 0.65) # Max 65% (approx 32px)
        elif char_type in descenders:
            target_height = int(safe_zone * 0.65) # Max 65% (approx 32px)
        elif char_type in short_letters:
            target_height = int(safe_zone * 0.35) # Small letters are tiny (~18px)
        else:
            target_height = int(safe_zone * 0.60)
            
        aspect = w / h
        new_h = target_height
        new_w = int(new_h * aspect)
        resized = cv2.resize(letter_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # --- POSITIONING ---
        norm_canvas = np.zeros((self.std_size, new_w), dtype=np.uint8)
        
        # FIX 3: LOWER BASELINE
        # We set the "floor" for letters like 'a' and 'b' at 70% down the box.
        # This leaves 30% of space at the bottom for descenders to hang into.
        baseline_y = int(self.std_size * 0.70) 
        
        if char_type in ['.', ',']:
            start_y = baseline_y - new_h
        elif char_type in ["'", '"', '`']:
            start_y = int(self.std_size * 0.15)
        elif char_type in descenders:
            # Logic: Top of 'y' should match Top of 'a'.
            # 'a' sits at baseline and is 0.35 tall.
            # So top of 'a' is: baseline - (safe_zone * 0.35)
            # We align 'y' to start there.
            short_letter_height = int(safe_zone * 0.35)
            x_height_top = baseline_y - short_letter_height
            start_y = x_height_top 
            # Note: Since 'y' is 0.65 tall, it will hang down to:
            # x_height_top + 0.65. This will fit perfectly in the remaining space.
        else:
            # Tall letters and Short letters sit on the baseline
            start_y = baseline_y - new_h

        # Strict Clamp (Just in case, but math above prevents it)
        if start_y < 0: start_y = 0
        if start_y + new_h > self.std_size: start_y = self.std_size - new_h

        norm_canvas[start_y:start_y+new_h, 0:new_w] = resized

        # --- INK FLOW SIMULATION ---
        kernel = np.ones((2, 2), np.uint8) 
        norm_canvas = cv2.dilate(norm_canvas, kernel, iterations=2) 
        _, norm_canvas = cv2.threshold(norm_canvas, 127, 255, cv2.THRESH_BINARY)

        rgba = np.zeros((self.std_size, new_w, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = self.ink_color
        rgba[:, :, 3] = norm_canvas
        
        return Image.fromarray(rgba)

    def generate(self, text, output_file="normalized_notes.png"):
        # 1. Create the Image Canvas
        canvas = Image.new('RGBA', (2480, 3508), (255, 255, 255, 255))
        
        # --- MAPPING & SETTINGS ---
        SPECIAL_CHAR_MAP = {
            '.': 'dot', '"': 'quote', ':': 'colon', '?': 'question',
            '*': 'asterisk', '/': 'slash', '\\': 'backslash',
            '<': 'lt', '>': 'gt', '|': 'pipe', ',': 'comma', "'": 'apostrophe'
        }
        
        start_x = 200
        max_x = 2200
        curr_x, curr_y = start_x, 200
        pixels_per_space = int(self.std_size * 0.4)
        
        # WIDTH FOR MISSING CHARACTERS
        # If a letter is missing, we leave a gap of this size (e.g., 25 pixels)
        fallback_width = int(self.std_size * 0.5)

        lines = text.split('\n')

        # --- DRAWING LOOP ---
        for line in lines:
            words = line.split(' ')
            
            for word in words:
                # We store tuples: (ImageObject, Width)
                # If image is missing, we store (None, fallback_width)
                word_items = [] 
                word_total_width = 0
                
                for char in word:
                    folder_name = SPECIAL_CHAR_MAP.get(char, char.lower())
                    char_dir = os.path.join(self.base_path, folder_name)
                    
                    found_image = False
                    
                    if os.path.isdir(char_dir):
                        variants = [f for f in os.listdir(char_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if variants:
                            img = self.process_letter_contour(
                                os.path.join(char_dir, random.choice(variants)), 
                                char
                            )
                            if img:
                                # SUCCESS: Add the letter image
                                width = img.width + self.char_spacing
                                word_items.append((img, width))
                                word_total_width += width
                                found_image = True
                    
                    # FAILURE: If folder missing, empty, or processing failed
                    if not found_image:
                        # Add a BLANK placeholder
                        # print(f"Missing: {char}") # Uncomment to debug
                        word_items.append((None, fallback_width))
                        word_total_width += fallback_width

                # Check if word fits on line
                if curr_x + word_total_width > max_x:
                    curr_x = start_x
                    curr_y += self.line_height

                # Draw the word
                for item in word_items:
                    img, width = item
                    
                    # Only paste if we actually have an image
                    if img is not None:
                        canvas.paste(img, (curr_x, curr_y), img)
                    
                    # ALWAYS move the cursor (creates the blank space if img is None)
                    curr_x += width

                curr_x += pixels_per_space
            
            curr_x = start_x
            curr_y += self.line_height

        # --- SAVING LOGIC ---
        if output_file.endswith(".docx"):
            temp_img_path = "temp_handwriting_render.png"
            canvas.convert("RGB").save(temp_img_path)
            
            doc = Document()
            section = doc.sections[0]
            section.left_margin = Cm(1.27)
            section.right_margin = Cm(1.27)
            section.top_margin = Cm(1.27)
            section.bottom_margin = Cm(1.27)
            
            doc.add_picture(temp_img_path, width=Inches(7.5))
            doc.save(output_file)
            os.remove(temp_img_path)
            print(f"Created Word Document: {output_file}")
            
        else:
            canvas.convert("RGB").save(output_file)
            print(f"Created Image: {output_file}")
# Usage
engine = NormalizedHandwritingEngine(ink_color=(0, 20, 100))
text = """I am applying for admission to the Master of Science in Robotics and Autonomous Systems at Khalifa University of Science and Technology, motivated by a strong interest in intelligent systems that integrate perception, data-driven decision-making, and real-world interaction. With an academic background in Computer Engineering and hands-on experience in machine learning, data analytics, and intelligent systems, I aim to develop advanced expertise in robotics and autonomy within a research-driven environment that supports the UAE’s vision for advanced technology and innovation.
Throughout my undergraduate studies, I was particularly drawn to problems that bridged software intelligence with physical and cyber-physical systems. My coursework and projects exposed me to the challenges of designing systems that sense, reason, and act in dynamic environments. This interest extended beyond theoretical learning into practical projects involving machine learning pipelines, real-time data processing, IoT-based systems, and intelligent decision frameworks, which shaped my ambition to pursue graduate training in robotics and autonomous systems.
I have worked on several projects that strengthened my foundation in data-driven intelligence, which I view as a critical component of modern autonomous systems. These include large-scale multilingual sentiment analysis, IoT-based data collection and analytics systems, and AI-driven control and decision-making architectures for smart environments. Through these projects, I gained experience in data preprocessing, feature extraction, model training, evaluation, and deployment under real-world constraints. These experiences highlighted the importance of robust perception, scalable computation, and reliable decision-making, all of which are central to autonomous robotic systems.
In addition to data-centric work, I have explored applications at the intersection of computer vision, real-time inference, and intelligent control, reinforcing my understanding that effective robotics systems require more than isolated algorithms. Successful autonomy depends on the integration of sensing, learning, control, and system-level design, as well as careful evaluation under physical and operational constraints. I am particularly interested in how machine learning and data-driven methods can enhance robotic perception, motion planning, and adaptive decision-making in autonomous systems.
The MSc in Robotics and Autonomous Systems curriculum at Khalifa University aligns strongly with my academic and professional goals. The program’s emphasis on robot perception, autonomous control, intelligent systems, and advanced AI techniques, combined with access to research facilities and interdisciplinary collaboration, makes it an ideal environment for my graduate studies. I am especially interested in engaging in research or thesis work related to autonomous robotics, intelligent sensing, and data-driven control, with applications in areas such as smart infrastructure, robotics for energy and sustainability, autonomous inspection, and intelligent urban systems.
Through this program, I aim to develop a strong theoretical foundation and practical expertise in robotics and autonomy, while deepening my skills in machine learning, statistical modeling, and scalable computation as they apply to real-world robotic systems. I am particularly interested in topics such as sensor fusion, learning-based control, data-efficient learning, and interpretable autonomous decision-making, which are critical for deploying reliable and safe robotic systems.
"""
text = text.lower()
engine.generate(text,output_file="my_homework.docx")