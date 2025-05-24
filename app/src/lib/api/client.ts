const API_BASE_URL = 'http://localhost:8000';

type RequestMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';

interface RequestOptions {
  method?: RequestMethod;
  body?: object;
  headers?: Record<string, string>;
}

async function apiRequest<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { method = 'GET', body, headers = {} } = options;
  
  const requestHeaders: HeadersInit = {
    'Content-Type': 'application/json',
    ...headers,
  };
  
  const config: RequestInit = {
    method,
    headers: requestHeaders,
    body: body ? JSON.stringify(body) : undefined,
  };
  
  const response = await fetch(`${API_BASE_URL}${endpoint}`, config);
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      `API error: ${response.status} ${response.statusText} - ${JSON.stringify(errorData)}`
    );
  }
  
  // For DELETE requests that don't return content
  if (method === 'DELETE') {
    return {} as T;
  }
  
  return response.json();
}

export const apiClient = {
  get: <T>(endpoint: string, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, { headers }),
    
  post: <T>(endpoint: string, data: object, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, { method: 'POST', body: data, headers }),
    
  put: <T>(endpoint: string, data: object, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, { method: 'PUT', body: data, headers }),
    
  delete: <T>(endpoint: string, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, { method: 'DELETE', headers }),
};