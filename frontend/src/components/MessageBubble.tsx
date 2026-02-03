import React, { useState } from "react";
import { Message } from "@/lib/types";
import SourceCard from "./SourceCard";

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4 animate-in fade-in slide-in-from-bottom-2 duration-300`}
    >
      <div className={`max-w-3xl ${isUser ? "ml-12" : "mr-12"}`}>
        {/* Message bubble */}
        <div
          className={`rounded-2xl px-5 py-3 ${
            isUser
              ? "bg-primary-600 text-white"
              : "bg-white border border-surface-200 text-surface-900"
          }`}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>

        {/* Timestamp */}
        <p className="text-xs text-surface-400 mt-1 px-2">
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>

        {/* Sources toggle button */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <button
            onClick={() => setShowSources(!showSources)}
            className="mt-2 px-3 py-1.5 text-xs font-medium text-primary-700 hover:text-primary-800 bg-primary-50 hover:bg-primary-100 rounded-lg transition-colors duration-150 flex items-center gap-1"
          >
            <svg
              className={`w-3 h-3 transition-transform duration-200 ${showSources ? "rotate-180" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
            {showSources ? "Hide" : "Show"} {message.sources.length} source
            {message.sources.length !== 1 ? "s" : ""}
          </button>
        )}

        {/* Sources panel */}
        {!isUser && showSources && message.sources && (
          <div className="mt-3 space-y-2">
            {message.sources.map((source, index) => (
              <SourceCard key={index} source={source} index={index} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
