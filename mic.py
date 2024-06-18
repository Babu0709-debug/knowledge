import streamlit as st
import cv2  # OpenCV for image processing

def capture_image():
  """Captures an image using the webcam and displays it."""
  cap = cv2.VideoCapture(0)  # Open default webcam (0)

  if not cap.isOpened():
    st.error("Error opening camera!")
    return

  # Capture image on button click
  if st.button("Capture Image"):
    ret, frame = cap.read()
    if ret:
      cv2.imwrite("captured_image.jpg", frame)
      st.success("Image captured successfully!")
      # Display captured image (optional)
      st.image("captured_image.jpg", use_column_width=True)
    else:
      st.error("Failed to capture image!")

  cap.release()  # Release camera resources

def main():
  """Main function for UI and image capture."""
  st.title("Image Capture (Streamlit Cloud)")

  capture_image()  # Call image capture function

if __name__ == "__main__":
  main()
