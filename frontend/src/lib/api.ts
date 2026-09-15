import { ApiError, type ApiErrorPayload } from "@/lib/errors";
import type { AskRequest, AskResponse, HealthResponse, UploadResponse } from "@/lib/types";

export const API_BASE_PATH = "/api";

export const endpoints = {
  health: `${API_BASE_PATH}/health`,
  upload: `${API_BASE_PATH}/upload`,
  query: `${API_BASE_PATH}/query`,
  ask: `${API_BASE_PATH}/ask`,
} as const;

async function parseJsonResponse<T>(response: Response, fallbackMessage: string): Promise<T> {
  const payload = (await readJson(response)) as T | ApiErrorPayload | null;

  if (!response.ok) {
    throw new ApiError(
      friendlyApiMessage(response.status, payload as ApiErrorPayload | null, fallbackMessage),
      response.status,
      payload as ApiErrorPayload | null,
    );
  }

  return payload as T;
}

async function readJson(response: Response): Promise<unknown | null> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function friendlyApiMessage(
  status: number,
  payload: ApiErrorPayload | null,
  fallbackMessage: string,
) {
  if (typeof payload?.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }

  if (Array.isArray(payload?.detail) && payload.detail.length > 0) {
    return "The request was not accepted. Check the inputs and try again.";
  }

  if (status === 0) {
    return "Backend unavailable. Confirm the FastAPI server is running on port 8000.";
  }

  return fallbackMessage;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(endpoints.health, {
    headers: {
      Accept: "application/json",
    },
  });

  return parseJsonResponse<HealthResponse>(response, "Could not reach the backend health check.");
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(endpoints.upload, {
    method: "POST",
    body: formData,
  });

  return parseJsonResponse<UploadResponse>(
    response,
    "Upload failed. Check the file and try again.",
  );
}

export async function askQuestion(request: AskRequest): Promise<AskResponse> {
  const response = await fetch(endpoints.ask, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  return parseJsonResponse<AskResponse>(
    response,
    "Answer generation failed. Try a narrower question or confirm Gemini is configured.",
  );
}
