"use client";

import React, { useState } from "react";
import FileUpload from "@/components/FileUpload";
import ChatInterface from "@/components/ChatInterface";
import Card, { CardHeader, CardBody } from "@/components/ui/Card";
import { useChat } from "@/hooks/useChat";

export default function Home() {
  const {
    messages,
    isLoading,
    error,
    documentLoaded,
    filename,
    sendMessage,
    clearChat,
    setDocumentLoaded,
    setError,
  } = useChat();

  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  const handleUploadSuccess = (filename: string, chunks: number) => {
    setDocumentLoaded(filename);
    setUploadSuccess(
      `Successfully uploaded "${filename}" (${chunks} chunks created)`
    );
    setUploadError(null);
    setTimeout(() => setUploadSuccess(null), 5000);
  };

  const handleUploadError = (error: string) => {
    setUploadError(error);
    setUploadSuccess(null);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-surface-200 px-6 py-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-surface-900">
                Document Chat
              </h1>
              <p className="text-sm text-surface-500">
                AI-powered document Q&A
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-7xl mx-auto px-6 py-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
            {/* Left sidebar - Upload */}
            <div className="lg:col-span-1 space-y-4">
              <Card>
                <CardHeader>
                  <h2 className="text-lg font-semibold text-surface-900">
                    Upload Document
                  </h2>
                  <p className="text-sm text-surface-500 mt-1">
                    Upload a document to start asking questions
                  </p>
                </CardHeader>
                <CardBody className="pt-0">
                  <FileUpload
                    onUploadSuccess={handleUploadSuccess}
                    onUploadError={handleUploadError}
                  />

                  {/* Upload feedback */}
                  {uploadSuccess && (
                    <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-start gap-2">
                        <svg
                          className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                            clipRule="evenodd"
                          />
                        </svg>
                        <p className="text-sm text-green-700">{uploadSuccess}</p>
                      </div>
                    </div>
                  )}

                  {uploadError && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-start gap-2">
                        <svg
                          className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                            clipRule="evenodd"
                          />
                        </svg>
                        <p className="text-sm text-red-700">{uploadError}</p>
                      </div>
                    </div>
                  )}
                </CardBody>
              </Card>

              {/* How it works */}
              <Card>
                <CardHeader>
                  <h2 className="text-lg font-semibold text-surface-900">
                    How it works
                  </h2>
                </CardHeader>
                <CardBody className="pt-0">
                  <ol className="space-y-3">
                    <li className="flex gap-3">
                      <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-primary-100 text-primary-700 rounded-full text-sm font-semibold">
                        1
                      </span>
                      <p className="text-sm text-surface-700">
                        Upload a PDF, TXT, or DOCX document
                      </p>
                    </li>
                    <li className="flex gap-3">
                      <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-primary-100 text-primary-700 rounded-full text-sm font-semibold">
                        2
                      </span>
                      <p className="text-sm text-surface-700">
                        Ask questions about the document content
                      </p>
                    </li>
                    <li className="flex gap-3">
                      <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-primary-100 text-primary-700 rounded-full text-sm font-semibold">
                        3
                      </span>
                      <p className="text-sm text-surface-700">
                        Get AI-powered answers with source citations
                      </p>
                    </li>
                  </ol>
                </CardBody>
              </Card>
            </div>

            {/* Right side - Chat */}
            <div className="lg:col-span-2">
              <Card className="h-full flex flex-col">
                <ChatInterface
                  messages={messages}
                  isLoading={isLoading}
                  error={error}
                  documentLoaded={documentLoaded}
                  filename={filename}
                  onSendMessage={sendMessage}
                  onClearChat={clearChat}
                />
              </Card>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-surface-200 px-6 py-4">
        <div className="max-w-7xl mx-auto text-center text-sm text-surface-500">
          Powered by Google Gemini AI • Built with Next.js & FastAPI
        </div>
      </footer>
    </div>
  );
}
