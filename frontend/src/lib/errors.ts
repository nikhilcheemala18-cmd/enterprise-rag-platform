export type ApiErrorPayload = {
  detail?: string | Array<Record<string, unknown>>;
};

export class ApiError extends Error {
  readonly status: number;
  readonly payload: ApiErrorPayload | null;

  constructor(message: string, status: number, payload: ApiErrorPayload | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    return error.message;
  }

  if (error instanceof TypeError) {
    return "Backend unavailable. Confirm the FastAPI server is running on port 8000.";
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  return fallback;
}
