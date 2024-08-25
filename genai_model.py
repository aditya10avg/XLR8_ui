#Following is the code for gen ai model 

"""import time
import torch
import cohere
from diffusers import DiffusionPipeline
import torchaudio
import ChatTTS"""

# Initialize Cohere client
api_key = "Veuz5kVIbxnNm3kdk5Kt3YTPrdFCZGfouoaxAsG7"
#co = cohere.Client(api_key)

# Initialize Diffusion pipeline globally
image_pipeline = None
video_pipeline = None
def generate_output(prompt):
    # Initialize ChatTTS
    chat = ChatTTS.Chat()
    chat.load_models(compile=False)  # Set to True for better performance

    def initialize_pipelines():
        global image_pipeline, video_pipeline
        try:
            image_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image_pipeline.to(device)  # Use GPU for faster processing if available
            print("Image Diffusion Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing Image DiffusionPipeline: {e}")

        try:
            video_pipeline = DiffusionPipeline.from_pretrained("ali-vilab/text-to-video-ms-1.7b")
            video_pipeline.to(device)
            print("Video Diffusion Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing Video DiffusionPipeline: {e}")

    # Function to generate follow-up questions
    def generate_follow_up_questions(input_text):
        initial_message = (
            f"You are a professional assistant designed to ask relevant follow-up questions to understand user requirements "
            f"for generating content such as images, audio, video etc from text prompts. The user input is '{input_text}', based on this generate "
            "three follow-up questions to gather more details about what the user wants and create an output close to their imagination "
            "Consider all the possible aspects to make it as close as possible to users imagination or need . "
            "I need just the questions in the Python list format with no other text."
        )
        response = co.generate(
            model='command-r-plus',
            prompt=initial_message,
            max_tokens=150,
            temperature=0.3,
        )
        questions = eval(response.generations[0].text.strip())
        return questions[:3]

    # Function to generate the final detailed prompt
    def generate_final_prompt(input_text, questions_and_answers):
        qna_text = "\n".join(questions_and_answers)
        prompt = (
            "State the purpose in just two  words like image generation, video generation converting image to 3d etc by analysing the initial prompt and the question answers and then generate a proper prompt for Gen AI models to generate required outputs.By analysing I just mean if it is related to video, audio or image etc followed by generation."
            " Based on the following original input and follow-up questions with answers"
            f"\n\nOriginal Input: {input_text}\n\nFollow-up Questions and Answers:\n{qna_text}"
        )
        response = co.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        final_prompt = response.generations[0].text.strip()
        return final_prompt

    # Function to identify the purpose of the final prompt
    def identify_purpose(final_prompt):
        first_line = final_prompt.strip().split('\n')[0].strip().lower()
        if "image generation" in first_line:
            return "image generation"
        elif "video generation" in first_line:
            return "video generation"
        elif "audio generation" in first_line:
            return "audio generation"
        else:
            return "unsupported purpose"

    # Function to generate the image
    def generate_image(detailed_prompt):
        try:
            if image_pipeline is None:
                initialize_pipelines()  # Initialize pipelines if not already initialized

            if image_pipeline is None:
                print("Error: DiffusionPipeline not initialized.")
                return None

            image = image_pipeline(detailed_prompt).images[0]
            output_path = "output.png"
            image.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error during image generation: {e}")
            return None

    # Function to generate the video
    def generate_video(detailed_prompt):
        try:
            if video_pipeline is None:
                initialize_pipelines()  # Initialize pipelines if not already initialized

            if video_pipeline is None:
                print("Error: Video DiffusionPipeline not initialized.")
                return None

            device = "cuda" if torch.cuda.is_available() else "cpu"
            video_pipeline.to(device)

            video = video_pipeline(detailed_prompt).videos[0]  # Hypothetical usage
            output_path = "/content/output.mp4"  # Adjust path for Colab
            video.save(output_path)
            return output_path
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error. Trying to allocate on CPU.")
                device = "cpu"
                video_pipeline.to(device)
                video = video_pipeline(detailed_prompt).videos[0]
                output_path = "/content/output.mp4"
                video.save(output_path)
                return output_path
            else:
                print(f"Error during video generation: {e}")
                return None
        except Exception as e:
            print(f"Error during video generation: {e}")
            return None

    # Function to generate the audio
    def generate_audio(text):
        try:
            wavs = chat.infer([text], sampling_rate=24000)  # Adjust sampling rate as needed
            output_path = "output.wav"
            torchaudio.save(output_path, torch.from_numpy(wavs[0]), 24000)
            return output_path
        except Exception as e:
            print(f"Error during audio generation: {e}")
            return None

    # Example user interaction
    def main():
        user_input = input("What do you want to generate: ")

        # Step 1: Generate follow-up questions
        follow_up_questions = generate_follow_up_questions(user_input)
        questions_and_answers = []

        # Step 2: Capture user's responses to follow-up questions
        for question in follow_up_questions:
            response = input(f"{question}: ")
            questions_and_answers.append(f"Question: {question} / Response: {response}")

        # Step 3: Generate the final prompt based on user input and Q&A
        final_prompt = generate_final_prompt(user_input, questions_and_answers)
        print(f"Final Prompt: {final_prompt}")

        # Step 4: Identify the purpose and route to the appropriate AI model
        purpose = identify_purpose(final_prompt)
        if purpose == "image generation":
            detailed_prompt_parts = final_prompt.split("\n", 1)
            detailed_prompt = detailed_prompt_parts[1] if len(detailed_prompt_parts) > 1 else ""
            output_path = generate_image(detailed_prompt.strip())
            if output_path:
                print(f"Generated image: {output_path}")
            else:
                print("Image generation failed.")
        elif purpose == "video generation":
            detailed_prompt_parts = final_prompt.split("\n", 1)
            detailed_prompt = detailed_prompt_parts[1] if len(detailed_prompt_parts) > 1 else ""
            output_path = generate_video(detailed_prompt.strip())
            if output_path:
                print(f"Generated video: {output_path}")
            else:
                print("Video generation failed.")
        elif purpose == "audio generation":
            output_path = generate_audio(user_input)
            if output_path:
                print(f"Generated audio: {output_path}")
            else:
                print("Audio generation failed.")
        else:
            print("Unsupported purpose")

    if __name__ == "__main__":
        main()
