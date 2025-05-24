import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { guideService } from '../api';
import type { CreateGuideRequest, CreateMessageRequest } from '../api';

export const useGuides = () => {
  return useQuery({
    queryKey: ['guides'],
    queryFn: () => guideService.listGuides(),
  });
};

export const useGuide = (id: string) => {
  return useQuery({
    queryKey: ['guides', id],
    queryFn: () => guideService.getGuide(id),
    enabled: !!id,
  });
};

export const useCreateGuide = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: CreateGuideRequest) => guideService.createGuide(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['guides'] });
    },
  });
};

export const useDeleteGuide = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => guideService.deleteGuide(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['guides'] });
    },
  });
};

export const useSendMessage = (guideId: string) => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: CreateMessageRequest) => 
      guideService.createMessage(guideId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['guides', guideId] });
    },
  });
};