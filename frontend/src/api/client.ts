import type { CartoonizeResponse, StyleOption, Variant } from "../types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.toString() ?? "http://localhost:8000";

async function handleJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Ignore parse errors and keep fallback message.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export async function fetchStyles(): Promise<StyleOption[]> {
  const response = await fetch(`${API_BASE_URL}/api/styles`);
  const payload = await handleJson<{ styles: StyleOption[] }>(response);
  return payload.styles;
}

export async function cartoonize(
  file: File,
  styleId: string,
  variant: Variant
): Promise<CartoonizeResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("style_id", styleId);
  form.append("variant", variant);

  const response = await fetch(`${API_BASE_URL}/api/cartoonize`, {
    method: "POST",
    body: form
  });
  return handleJson<CartoonizeResponse>(response);
}

export function resolveResultUrl(path: string): string {
  return new URL(path, API_BASE_URL).toString();
}
