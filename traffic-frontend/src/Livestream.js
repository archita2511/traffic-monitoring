import React, { useState } from "react";
import axios from "axios";

export default function LiveStream() {
  const [videoFile, setVideoFile] = useState(null);
  const [showStream, setShowStream] = useState(false);

  const uploadAndStream = async () => {
    const formData = new FormData();
    formData.append("video", videoFile);
    await axios.post("http://localhost:5000/upload", formData);
    setShowStream(true);
  };

  return (
    <div>
      <h2>Vehicle Speed Detection (Live)</h2>
      <input type="file" accept="video/*" onChange={e => setVideoFile(e.target.files[0])} />
      <button onClick={uploadAndStream}>Upload & Start Stream</button>

      {showStream && (
        <div style={{ marginTop: 20 }}>
          <img src="http://localhost:5000/video_feed" alt="live stream" />
        </div>
      )}
    </div>
  );
}
