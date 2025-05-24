import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { assetService } from '../api';
import type { CreateAssetRequest } from '../api';

export const useAssets = () => {
  return useQuery({
    queryKey: ['assets'],
    queryFn: () => assetService.listAssets(),
  });
};

export const useAsset = (id: string) => {
  return useQuery({
    queryKey: ['assets', id],
    queryFn: () => assetService.getAsset(id),
    enabled: !!id,
  });
};

export const useCreateAsset = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: CreateAssetRequest) => assetService.createAsset(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['assets'] });
    },
  });
};

export const useDeleteAsset = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => assetService.deleteAsset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['assets'] });
    },
  });
};