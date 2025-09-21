import { useState, useCallback, useRef, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Upload, FileVideo, X, Play, Download } from "lucide-react";

interface VideoUploadProps {
  onVideoProcessed?: (videoUrl: string) => void;
}

type ProcessingStage = "idle" | "uploading" | "analyzing" | "tracking" | "generating" | "complete";

const VideoUpload = ({ onVideoProcessed }: VideoUploadProps) => {
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string>("");
  const [processingStage, setProcessingStage] = useState<ProcessingStage>("idle");
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  // Ref-based preview assignment avoids dynamic src interpolation flagged by CodeQL
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const prevUrlsRef = useRef<{ preview?: string; processed?: string }>({});

  // Safely assign the video preview source only for blob: URLs and revoke previous URL objects
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    const prev = prevUrlsRef.current.preview;
    if (videoUrl && videoUrl.startsWith("blob:")) {
      el.src = videoUrl; // safe: created via URL.createObjectURL(File)
      // Revoke previous blob URL if different
      if (prev && prev !== videoUrl && prev.startsWith("blob:")) {
        try { URL.revokeObjectURL(prev); } catch {}
      }
      prevUrlsRef.current.preview = videoUrl;
    } else {
      // Clear any non-blob or empty value
      el.removeAttribute("src");
      el.load();
    }
  }, [videoUrl]);

  // Revoke old processed blob URLs when replaced
  useEffect(() => {
    const prev = prevUrlsRef.current.processed;
    if (processedVideoUrl && processedVideoUrl.startsWith("blob:")) {
      if (prev && prev !== processedVideoUrl && prev.startsWith("blob:")) {
        try { URL.revokeObjectURL(prev); } catch {}
      }
      prevUrlsRef.current.processed = processedVideoUrl;
    }
  }, [processedVideoUrl]);

  const stageMessages = {
    idle: "Ready to analyze",
    uploading: "Uploading video...",
    analyzing: "Detecting court boundaries...", 
    tracking: "Tracking players and ball...",
    generating: "Generating analysis video...",
    complete: "Analysis complete!"
  };

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const videoFile = files.find(file => file.type.startsWith('video/'));
    
    if (videoFile) {
      setUploadedVideo(videoFile);
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      toast({
        title: "Video uploaded successfully",
        description: `${videoFile.name} is ready for analysis`,
      });
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload a video file",
        variant: "destructive",
      });
    }
  }, [toast]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedVideo(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      toast({
        title: "Video uploaded successfully", 
        description: `${file.name} is ready for analysis`,
      });
    }
  };

  const startProcessing = async () => {
    if (!uploadedVideo) {
      toast({ title: "No video", description: "Please upload a video first", variant: "destructive" });
      return;
    }
    try {
      setProcessingStage("uploading");
      setProgress(5);

      const form = new FormData();
      form.append("video", uploadedVideo);

      // The dev proxy in vite.config.ts routes /api to http://localhost:8000
      const res = await fetch("/api/process", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed: ${res.status}`);
      }
      setProcessingStage("generating");
      setProgress(85);

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setProcessedVideoUrl(url);
      setProcessingStage("complete");
      setProgress(100);
      onVideoProcessed?.(url);
      toast({ title: "Analysis complete!", description: "Your tennis video has been analyzed successfully" });
    } catch (e: any) {
      console.error(e);
      toast({ title: "Processing failed", description: e?.message || "Unknown error", variant: "destructive" });
      setProcessingStage("idle");
      setProgress(0);
    }
  };

  const clearVideo = () => {
    setUploadedVideo(null);
    // Revoke any active blob URLs
    try {
      if (videoUrl && videoUrl.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    } catch {}
    try {
      if (processedVideoUrl && processedVideoUrl.startsWith("blob:")) URL.revokeObjectURL(processedVideoUrl);
    } catch {}
    setVideoUrl("");
    setProcessedVideoUrl("");
    setProcessingStage("idle");
    setProgress(0);
  };

  const downloadVideo = () => {
    if (processedVideoUrl && processedVideoUrl.startsWith("blob:")) {
      const a = document.createElement('a');
      a.href = processedVideoUrl; // codeql[js/xss-through-dom] false positive: constrained to blob: URL.createObjectURL
      a.download = `${uploadedVideo?.name?.replace(/\.[^/.]+$/, "")}_analyzed.mp4` || 'analyzed_video.mp4';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      
      toast({
        title: "Download started",
        description: "Your analyzed video is being downloaded",
      });
    }
  };

  return (
    <div className="space-y-6">
      {!uploadedVideo ? (
        <Card className="border-2 border-dashed border-muted-foreground/25 hover:border-primary/50 transition-colors">
          <CardContent className="p-8">
            <div
              className="flex flex-col items-center justify-center space-y-4 text-center"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="w-8 h-8 text-primary" />
              </div>
              
              <div className="space-y-2">
                <h3 className="text-xl font-semibold">Upload Tennis Video</h3>
                <p className="text-muted-foreground">
                  Drag and drop your video file here, or click to select
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports MP4, MOV, AVI formats
                </p>
              </div>

              <label htmlFor="video-upload">
                <Button variant="outline" className="cursor-pointer" asChild>
                  <span>
                    <FileVideo className="w-4 h-4 mr-2" />
                    Choose Video File
                  </span>
                </Button>
              </label>
              
              <input
                id="video-upload"
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {/* Video Preview */}
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <FileVideo className="w-5 h-5 text-primary" />
                  <span className="font-medium">{uploadedVideo.name}</span>
                </div>
                <Button variant="ghost" size="sm" onClick={clearVideo}>
                  <X className="w-4 h-4" />
                </Button>
              </div>

              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  controls
                  className="w-full h-full object-contain"
                />
              </div>
            </CardContent>
          </Card>

          {/* Processing Status */}
          {processingStage !== "idle" && (
            <Card>
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{stageMessages[processingStage]}</span>
                    <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            {processingStage === "idle" && (
              <Button 
                onClick={startProcessing} 
                className="flex-1"
              >
                <Play className="w-4 h-4 mr-2" />
                Start Analysis
              </Button>
            )}
            
            {processingStage === "complete" && processedVideoUrl && (
              <Button 
                onClick={downloadVideo}
                variant="outline"
                className="flex-1"
              >
                <Download className="w-4 h-4 mr-2" />
                Download Analyzed Video
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;