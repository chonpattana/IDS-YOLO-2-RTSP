#ffmpeg-video01.ipynb
import cv2
import subprocess

# Parameters
input_width = 640
input_height = 480
output_width = 320
#output_width = 170
output_height = 240 
#output_height = 120
output_fps = 30
output_file = 'output_compressed.mp4'
crf = 30  # Constant Rate Factor for compression quality

# OpenCV VideoCapture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

# FFmpeg command to compress video frames
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output file if exists
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{output_width}x{output_height}',
    '-pix_fmt', 'bgr24',
    '-r', str(output_fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-crf', str(crf),
    output_file
]

# Start FFmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame
    frame = cv2.resize(frame, (output_width, output_height))
    
    # Write frame to FFmpeg process
    ffmpeg_process.stdin.write(frame.tobytes())

    # Display the compressed frame
    cv2.imshow('Compressed Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()
