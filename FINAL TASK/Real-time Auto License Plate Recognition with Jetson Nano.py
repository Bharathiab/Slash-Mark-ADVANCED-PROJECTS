import cv2
import pytesseract

# Load pre-trained Haar Cascade model for license plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in plates:
        # Draw rectangle around the detected license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop the detected plate
        plate_image = gray[y:y + h, x:x + w]
        
        # Use Tesseract to do OCR on the plate
        plate_text = pytesseract.image_to_string(plate_image, config='--psm 8')
        print(f"Detected License Plate: {plate_text.strip()}")
        
        # Optionally, display the detected plate text
        cv2.putText(frame, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('License Plate Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
