To run the YOLO version:
1. To get the test video, download the test videos from the following link: https://drive.google.com/drive/folders/1WAJ-ndMIdPIHKP9wXxGXvYcfJAlwOZLk?usp=sharing
2. Navigate to traffic-backend folder and run 'python app.py
3. Navigate to traffic-frontend folder and run 'pm start'
4. Frontend would run on port 3000 backend would run on port 5000
5. Upload the .mp4 video using 'Choose file' option. Click on Upload and stream
6. You can view the video with the count of vehicles in either of the directions, speed of each vehicle, overspeeding alert, etc.,
7. Please re-run the application for every test video


To run Faster RCNN or SSD versions:
1. Navigate to traffic-backend folder
2. To get the test video, download the test videos from the following link and upload it to traffic-backend folder: https://drive.google.com/drive/folders/1WAJ-ndMIdPIHKP9wXxGXvYcfJAlwOZLk?usp=sharing
3. run main_rcnn_and_ssd by passing appropriate parameters
python main_rcnn_and_ssd.py --model rcnn --video path/to/test1.mp4
or
python main.py --model ssd --video path/to/test1.mp4
4. You can view the video with the count of vehicles in either of the directions, speed of each vehicle, etc.,
5. For RCNN, the processing time is huge. So it is recommended to wait until the video is processed before looking at the output video.