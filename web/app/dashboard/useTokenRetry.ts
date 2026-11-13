"use client";
import { useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import { AuthError } from "@/lib/api";
export function useTokenRetry() {
  const { getToken, logout, user } = useAuth();
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
          // Only a signed-in user's rejected token means the session is over; before Firebase has
          // restored the sign-in there is simply no token yet.
          if (retry instanceof AuthError && user) {
            await logout();
            window.location.assign("/login");
          }
          throw retry;
        }
      }
    },
    [getToken, logout, user],
  );
}
