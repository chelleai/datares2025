import { apiClient } from '../client';
import {
  Asset,
  CreateAssetRequest,
  CreateAssetResponse,
  ListAssetsResponse
} from '../types';

const ASSETS_ENDPOINT = '/assets';

export const assetService = {
  listAssets: () => 
    apiClient.get<ListAssetsResponse>(ASSETS_ENDPOINT),
  
  getAsset: (id: string) => 
    apiClient.get<Asset>(`${ASSETS_ENDPOINT}/${id}`),
  
  createAsset: (data: CreateAssetRequest) => 
    apiClient.post<CreateAssetResponse>(ASSETS_ENDPOINT, data),
  
  deleteAsset: (id: string) => 
    apiClient.delete(`${ASSETS_ENDPOINT}/${id}`)
};