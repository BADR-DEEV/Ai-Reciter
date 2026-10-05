// Off unless web/.env.local sets NEXT_PUBLIC_DEV_MODE=1. Public so the header can show the link.
export const DEV_MODE = ["1", "true", "on"].includes((process.env.NEXT_PUBLIC_DEV_MODE || "").toLowerCase());
