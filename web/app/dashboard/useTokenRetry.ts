"use client";
import { useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import { AuthError } from "@/lib/api";
export function useTokenRetry() {
  const { getToken, logout } = useAuth();
  return useCallback(
    async <T>(operation: () => Promise<T>): Promise<T> => {
      try {
        if (!(await getToken())) throw new AuthError();
        return await operation();
      } catch (e) {
        if (!(e instanceof AuthError)) throw e;
        try {
          if (!(await getToken(true))) throw new AuthError();
          return await operation();
        } catch (retry) {
          if (retry instanceof AuthError) {
            await logout();
            window.location.assign("/login");
          }
          throw retry;
        }
      }
    },
    [getToken, logout],
  );
}
