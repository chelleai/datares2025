import { apiClient } from '../client';
import {
  CreateGuideRequest,
  CreateGuideResponse,
  CreateMessageRequest,
  CreateMessageResponse,
  Guide,
  ListGuidesResponse
} from '../types';

const GUIDES_ENDPOINT = '/guides';

export const guideService = {
  listGuides: () => 
    apiClient.get<ListGuidesResponse>(GUIDES_ENDPOINT),
  
  getGuide: (id: string) => 
    apiClient.get<Guide>(`${GUIDES_ENDPOINT}/${id}`),
  
  createGuide: (data: CreateGuideRequest) => 
    apiClient.post<CreateGuideResponse>(GUIDES_ENDPOINT, data),
  
  createMessage: (guideId: string, data: CreateMessageRequest) => 
    apiClient.post<CreateMessageResponse>(
      `${GUIDES_ENDPOINT}/${guideId}/messages`, 
      data
    ),
  
  deleteGuide: (id: string) => 
    apiClient.delete(`${GUIDES_ENDPOINT}/${id}`)
};