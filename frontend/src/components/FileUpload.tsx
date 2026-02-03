"use client";

import React, { useState, useRef } from "react";
import Button from "./ui/Button";
import Card, { CardBody } from "./ui/Card";
import { api } from "@/lib/api";

interface FileUploadProps {
  onUploadSuccess: (filename: string, chunks: number) => void;
  onUploadError: (error: string) => void;
}

export default function FileUpload({
  onUploadSuccess,
  onUploadError,
}: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptedFormats = [".pdf", ".txt", ".docx"];
  const maxSizeMB = 10;

  const handleFileSelect = (file: File) => {
    // Validate file type
    const extension = "." + file.name.split(".").pop()?.toLowerCase();
    if (!acceptedFormats.includes(extension)) {
      onUploadError(
        `Unsupported file type. Accepted formats: ${acceptedFormats.join(", ")}`
      );
      return;
    }

    // Validate file size
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > maxSizeMB) {
      onUploadError(`File too large. Maximum size: ${maxSizeMB}MB`);
      return;
    }

    setSelectedFile(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    try {
      const response = await api.uploadDocument(selectedFile);
      onUploadSuccess(response.filename, response.total_chunks);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      onUploadError(
        error instanceof Error ? error.message : "Failed to upload document"
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Card>
      <CardBody>
        <div className="space-y-4">
          {/* Drag and drop area */}
          <div
            className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? "border-primary-500 bg-primary-50"
                : "border-surface-300 hover:border-surface-400"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept={acceptedFormats.join(",")}
              onChange={handleFileInputChange}
              disabled={isUploading}
            />

            <div className="space-y-3">
              {/* Upload icon */}
              <div className="flex justify-center">
                <svg
                  className="w-12 h-12 text-surface-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>

              {selectedFile ? (
                <div>
                  <p className="text-sm font-medium text-surface-900">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-surface-500">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-sm font-medium text-surface-900">
                    Drop your document here, or{" "}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="text-primary-600 hover:text-primary-700 font-semibold"
                      disabled={isUploading}
                    >
                      browse
                    </button>
                  </p>
                  <p className="text-xs text-surface-500 mt-1">
                    Supported formats: PDF, TXT, DOCX (max {maxSizeMB}MB)
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Upload button */}
          {selectedFile && (
            <div className="flex gap-2">
              <Button
                variant="primary"
                onClick={handleUpload}
                isLoading={isUploading}
                disabled={isUploading}
                className="flex-1"
              >
                {isUploading ? "Uploading..." : "Upload & Process"}
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  setSelectedFile(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
                disabled={isUploading}
              >
                Cancel
              </Button>
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  );
}
