import React, { useState } from "react";
import axios from "axios";
import { Button, Typography, Box, Stack } from "@mui/material";

export default function LiveStream() {
  const [videoFile, setVideoFile] = useState(null);
  const [showStream, setShowStream] = useState(false);
  const [loading, setLoading] = useState(false);

  const uploadAndStream = async () => {
    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      setLoading(true);
      await axios.post("http://localhost:5000/upload", formData);
      setShowStream(true);
    } catch (error) {
      alert("Upload failed or server error.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box textAlign="center" py={5}>
      <Typography variant="h4" gutterBottom>
        🚗 AI Traffic Monitoring
      </Typography>

      <Stack direction="row" justifyContent="center" spacing={2} alignItems="center" mb={4}>
        <input
          type="file"
          accept="video/*"
          id="video-upload"
          style={{ display: "none" }}
          onChange={(e) => setVideoFile(e.target.files[0])}
        />
        <label htmlFor="video-upload">
          <Button variant="contained" component="span" color="secondary">
            {videoFile ? videoFile.name : "Choose File"}
          </Button>
        </label>

        <Button
          variant="contained"
          onClick={uploadAndStream}
          disabled={!videoFile || loading}
          color="primary"
        >
          {loading ? "Processing..." : "Upload & Stream"}
        </Button>
      </Stack>

      {showStream && (
        <Box mt={4}>
          <img
            src="http://localhost:5000/video_feed"
            alt="Video stream"
            style={{
              width: "80%",
              maxWidth: "900px",
              border: "1px solid #ccc",
              borderRadius: "8px",
            }}
          />
        </Box>
      )}
    </Box>
  );
}