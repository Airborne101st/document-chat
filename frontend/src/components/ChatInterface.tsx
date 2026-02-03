"use client";

import React, { useState, useRef, useEffect } from "react";
import Button from "./ui/Button";
import MessageBubble from "./MessageBubble";
import Spinner from "./ui/Spinner";
import { Message } from "@/lib/types";

interface ChatInterfaceProps {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  documentLoaded: boolean;
  filename: string | null;
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
}

export default function ChatInterface({
  messages,
  isLoading,
  error,
  documentLoaded,
  filename,
  onSendMessage,
  onClearChat,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200 bg-white">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${documentLoaded ? "bg-green-500" : "bg-surface-300"}`}
            />
            <span className="text-sm font-medium text-surface-700">
              {documentLoaded ? filename : "No document loaded"}
            </span>
          </div>
        </div>
        {messages.length > 0 && (
          <Button variant="secondary" onClick={onClearChat} disabled={isLoading}>
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Clear
          </Button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 bg-surface-50">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 mb-4 rounded-full bg-primary-100 flex items-center justify-center">
              <svg
                className="w-8 h-8 text-primary-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-surface-900 mb-2">
              {documentLoaded
                ? "Ready to answer questions"
                : "Upload a document to get started"}
            </h3>
            <p className="text-sm text-surface-500 max-w-md">
              {documentLoaded
                ? `Ask questions about "${filename}" and get AI-powered answers with source citations.`
                : "Upload a PDF, TXT, or DOCX file to start asking questions."}
            </p>
          </div>
        ) : (
          <div>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-white border border-surface-200 rounded-2xl px-5 py-3">
                  <Spinner size="sm" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="px-6 py-3 bg-red-50 border-t border-red-200">
          <div className="flex items-center gap-2 text-sm text-red-700">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            {error}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="px-6 py-4 border-t border-surface-200 bg-white">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                documentLoaded
                  ? "Ask a question about your document..."
                  : "Upload a document first"
              }
              disabled={!documentLoaded || isLoading}
              className="input resize-none min-h-[52px] max-h-[200px]"
              rows={1}
            />
          </div>
          <Button
            type="submit"
            variant="primary"
            disabled={!input.trim() || !documentLoaded || isLoading}
            className="self-end"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </Button>
        </form>
        <p className="text-xs text-surface-500 mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
