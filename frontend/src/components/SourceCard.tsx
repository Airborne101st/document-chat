import React from "react";
import { Source } from "@/lib/types";
import Card from "./ui/Card";

interface SourceCardProps {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: SourceCardProps) {
  return (
    <Card className="mb-3">
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-6 h-6 text-xs font-semibold text-primary-700 bg-primary-100 rounded-full">
              {index + 1}
            </span>
            <span className="text-sm font-medium text-surface-700">
              {source.filename}
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-surface-500">
            <span>Page {source.page_number}</span>
            <span className="inline-flex items-center gap-1">
              <svg
                className="w-3 h-3"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              {(source.relevance_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <p className="text-sm text-surface-600 leading-relaxed line-clamp-3">
          {source.content}
        </p>
      </div>
    </Card>
  );
}
