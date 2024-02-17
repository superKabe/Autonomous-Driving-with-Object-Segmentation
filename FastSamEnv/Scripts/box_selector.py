import cv2


def select_object(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Initialize variables for box selection
    bbox = (0, 0, 0, 0)
    selecting = False

    # Callback function for mouse events
    def draw_box(event, x, y, flags, param):
        nonlocal selecting, bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = (x, y, 0, 0)
            selecting = True

        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            print("Selected Box Coordinates (x, y, width, height):", bbox)

        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            # Update the width and height of the bounding box while dragging
            bbox = (bbox[0], bbox[1], x - bbox[0], y - bbox[1])

    # Create a window and set the callback function
    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", draw_box)

    while True:
        # Copy the original image to draw on
        img_copy = image.copy()

        # Draw the box dynamically while selecting
        if selecting:
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            # Display the coordinates on the image
            cv2.putText(img_copy, f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the image with the selected box and coordinates
        cv2.imshow("Select Object", img_copy)

        # Break the loop if 'Enter' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break

    # Destroy the window and return the selected box coordinates
    cv2.destroyAllWindows()
    return bbox


if __name__ == '__main__':
    image_path = r"C:\MECE Project\Road Detection Python Project\FastSamEnv\FastSAM\images\cat.jpg"
    select_object(image_path)
