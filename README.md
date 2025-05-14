To run the YOLO version:
1. Download appropriate video of traffic
2. Navigate to traffic-backend folder and run 'python app.py
3. Navigate to traffic-frontend folder and run 'pm start'
4. Frontend would run on port 3000 backend would run on port 5000
5. Upload the .mp4 video using 'Choose file' option. Click on Upload and stream
6. You can view the video with the count of vehicles in either of the directions, speed of each vehicle, overspeeding alert, etc.,


To run Faster RCNN or SSD versions:
1. Navigate to traffic-backend folder
2. run main_rcnn_and_ssd by passing appropriate parameters
python main.py --model rcnn --video path/to/video.mp4
or
python main.py --model ssd --video path/to/video.mp4
3. You can view the video with the count of vehicles in either of the directions, speed of each vehicle, etc.,

   