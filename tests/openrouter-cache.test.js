import { describe, expect, test } from '@jest/globals';

import { deriveOpenRouterSessionId } from '../src/endpoints/backends/openrouter-cache.js';

describe('deriveOpenRouterSessionId', () => {
    test('returns a stable opaque value within OpenRouter limits', () => {
        const first = deriveOpenRouterSessionId('server secret', 'user', 'chat.jsonl');
        const second = deriveOpenRouterSessionId('server secret', 'user', 'chat.jsonl');

        expect(first).toBe(second);
        expect(first).toMatch(/^[a-f0-9]{32}$/);
        expect(first).not.toContain('chat');
    });

    test('scopes the value by local user and chat', () => {
        const original = deriveOpenRouterSessionId('server secret', 'user', 'chat.jsonl');

        expect(deriveOpenRouterSessionId('server secret', 'other-user', 'chat.jsonl')).not.toBe(original);
        expect(deriveOpenRouterSessionId('server secret', 'user', 'other-chat.jsonl')).not.toBe(original);
    });
});
