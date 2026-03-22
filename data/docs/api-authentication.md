---
tags: [api, authentication, security]
---

# API Authentication Guide

## Overview

Our API uses token-based authentication via OAuth2. All requests to protected
endpoints must include a valid bearer token in the Authorization header.

## Getting Started

### Step 1: Create an API key

Navigate to Settings > API Keys in the dashboard. Click "Generate New Key" and
store the key securely — it will only be shown once.

### Step 2: Exchange for a token

```bash
curl -X POST https://api.example.com/auth/token \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_KEY" \
  -d "client_secret=YOUR_SECRET"
```

The response contains an `access_token` valid for 1 hour.

### Step 3: Use the token

Include the token in all subsequent requests:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.example.com/v1/documents
```

## Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| 401  | Invalid or expired token | Re-authenticate using Step 2 |
| 403  | Insufficient permissions | Check your API key's role assignments |
| 429  | Rate limit exceeded | Wait and retry with exponential backoff |

## Rate Limits

- Standard tier: 100 requests/minute
- Pro tier: 1,000 requests/minute
- Enterprise tier: Custom limits

Rate limit headers are included in every response:
- `X-RateLimit-Limit`: Your tier's limit
- `X-RateLimit-Remaining`: Requests left in the current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

## Token Refresh

Tokens expire after 1 hour. Implement proactive refresh by checking the
`exp` claim in the JWT payload. Refresh 5 minutes before expiration to
avoid service interruptions.

## Security Best Practices

- Never commit API keys to version control
- Rotate keys every 90 days
- Use environment variables for key storage
- Enable IP allowlisting for production keys
- Monitor the audit log for unusual authentication patterns
