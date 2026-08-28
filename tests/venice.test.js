import { describe, expect, test } from '@jest/globals';

import {
    buildVeniceParameters,
    deriveVenicePromptCacheKey,
    normalizeVeniceModels,
} from '../src/endpoints/backends/venice.js';

describe('Venice provider helpers', () => {
    test('normalizes available text models and preserves capability metadata', () => {
        const payload = {
            data: [
                {
                    id: 'text-model',
                    type: 'text',
                    model_spec: {
                        availableContextTokens: 131072,
                        maxCompletionTokens: 8192,
                        capabilities: { supportsVision: true },
                    },
                },
                { id: 'offline-model', type: 'text', model_spec: { offline: true } },
                { id: 'image-model', type: 'image', model_spec: {} },
            ],
        };

        expect(normalizeVeniceModels(payload, { default: 'text-model' })).toEqual([
            expect.objectContaining({
                id: 'default',
                canonical_id: 'text-model',
                context_length: 131072,
                model_spec: expect.objectContaining({ name: 'Venice Default' }),
            }),
            expect.objectContaining({
                id: 'text-model',
                context_length: 131072,
                max_completion_tokens: 8192,
                model_spec: expect.objectContaining({ capabilities: { supportsVision: true } }),
            }),
        ]);
    });

    test('maps supported Venice request parameters', () => {
        expect(buildVeniceParameters({
            venice_character_slug: ' alan-watts ',
            venice_strip_thinking_response: true,
            venice_disable_thinking: true,
            venice_web_search: 'auto',
            venice_enable_web_scraping: true,
            venice_enable_web_citations: true,
            venice_include_system_prompt: false,
            venice_enable_x_search: true,
        })).toEqual({
            character_slug: 'alan-watts',
            strip_thinking_response: true,
            disable_thinking: true,
            enable_web_search: 'auto',
            enable_web_scraping: true,
            enable_web_citations: true,
            include_venice_system_prompt: false,
            enable_x_search: true,
        });
    });

    test('derives the default alias from model metadata when trait discovery fails', () => {
        const payload = {
            data: [{
                id: 'fallback-default',
                type: 'text',
                model_spec: { traits: ['default'], capabilities: { supportsFunctionCalling: true } },
            }],
        };

        expect(normalizeVeniceModels(payload)[0]).toEqual(expect.objectContaining({
            id: 'default',
            canonical_id: 'fallback-default',
            model_spec: expect.objectContaining({ capabilities: { supportsFunctionCalling: true } }),
        }));
    });

    test('uses documented Venice defaults for invalid or omitted options', () => {
        expect(buildVeniceParameters({ venice_web_search: 'invalid' })).toEqual({
            character_slug: undefined,
            strip_thinking_response: false,
            disable_thinking: false,
            enable_web_search: 'off',
            enable_web_scraping: false,
            enable_web_citations: false,
            include_venice_system_prompt: false,
            enable_x_search: false,
        });
    });

    test('derives stable user- and chat-scoped opaque cache keys', () => {
        const key = deriveVenicePromptCacheKey('server secret', 'user', 'chat.jsonl');

        expect(key).toMatch(/^[a-f0-9]{32}$/);
        expect(key).toBe(deriveVenicePromptCacheKey('server secret', 'user', 'chat.jsonl'));
        expect(key).not.toBe(deriveVenicePromptCacheKey('server secret', 'other-user', 'chat.jsonl'));
        expect(key).not.toBe(deriveVenicePromptCacheKey('server secret', 'user', 'other-chat.jsonl'));
    });
});
