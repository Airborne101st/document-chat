"use client";

import { useState, useCallback, useRef } from "react";
import { Message, ChatState } from "@/lib/types";
import { api } from "@/lib/api";

export function useChat() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    error: null,
    documentLoaded: false,
    filename: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const setDocumentLoaded = useCallback((filename: string) => {
    setState((prev) => ({
      ...prev,
      documentLoaded: true,
      filename,
      messages: [],
      error: null,
    }));
  }, []);

  const addMessage = useCallback((message: Message) => {
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
  }, []);

  const updateLastMessage = useCallback((updates: Partial<Message>) => {
    setState((prev) => ({
      ...prev,
      messages: prev.messages.map((msg, index) =>
        index === prev.messages.length - 1 ? { ...msg, ...updates } : msg
      ),
    }));
  }, []);

  const setError = useCallback((error: string | null) => {
    setState((prev) => ({
      ...prev,
      error,
      isLoading: false,
    }));
  }, []);

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim() || state.isLoading) return;

      // Add user message
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: query,
        timestamp: new Date(),
      };

      addMessage(userMessage);

      // Create assistant message placeholder
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };

      addMessage(assistantMessage);

      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        let fullAnswer = "";
        let sources = undefined;

        // Use streaming API
        for await (const event of api.queryStream({ query })) {
          if (event.type === "token") {
            fullAnswer += event.data;
            updateLastMessage({ content: fullAnswer });
          } else if (event.type === "sources") {
            sources = event.data;
            updateLastMessage({ sources });
          } else if (event.type === "done") {
            setState((prev) => ({ ...prev, isLoading: false }));
          } else if (event.type === "error") {
            setError(event.data);
            // Remove the empty assistant message on error
            setState((prev) => ({
              ...prev,
              messages: prev.messages.slice(0, -1),
            }));
            return;
          }
        }

        setState((prev) => ({ ...prev, isLoading: false }));
      } catch (error) {
        setError(error instanceof Error ? error.message : "Failed to send message");
        // Remove the empty assistant message on error
        setState((prev) => ({
          ...prev,
          messages: prev.messages.slice(0, -1),
        }));
      }
    },
    [state.isLoading, addMessage, updateLastMessage, setError]
  );

  const clearChat = useCallback(() => {
    setState((prev) => ({
      ...prev,
      messages: [],
      error: null,
    }));
  }, []);

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    error: state.error,
    documentLoaded: state.documentLoaded,
    filename: state.filename,
    sendMessage,
    clearChat,
    setDocumentLoaded,
    setError,
  };
}
