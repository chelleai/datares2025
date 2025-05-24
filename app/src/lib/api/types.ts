// Asset Types
export interface Concept {
  term: string;
  citations: string[];
  definition: string;
}

export interface Asset {
  id: string;
  name: string;
  content: string;
  concepts: Concept[];
}

export interface AssetOverview {
  id: string;
  name: string;
  concepts: Concept[];
}

export interface CreateAssetRequest {
  name: string;
  content: string;
}

export interface CreateAssetResponse {
  id: string;
}

export interface ListAssetsResponse {
  assets: AssetOverview[];
}

// Guide Types
export interface GuideConcept {
  term: string;
  definition: string;
}

export interface Guide {
  id: string;
  name: string;
  concepts: GuideConcept[];
  student_learning_style: string;
  user_messages: any[]; // Using any for now as we don't have the exact type
  assistant_messages: any[]; // Using any for now as we don't have the exact type
}

export interface GuideOverview {
  id: string;
  name: string;
  concepts: GuideConcept[];
}

export interface CreateGuideRequest {
  name: string;
  concepts: GuideConcept[];
  student_learning_style: string;
}

export interface CreateGuideResponse {
  id: string;
}

export interface CreateMessageRequest {
  message: string;
}

export interface CreateMessageResponse {
  message: string;
}

export interface ListGuidesResponse {
  guides: GuideOverview[];
}