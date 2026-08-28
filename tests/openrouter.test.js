import { describe, expect, test } from '@jest/globals';

process.env.SILLYTAVERN_ALLOWKEYSEXPOSURE = 'false';
const { getOpenRouterProviderNames } = await import('../src/endpoints/openrouter.js');

describe('getOpenRouterProviderNames', () => {
    test('extracts unique providers for a selected model', () => {
        const response = {
            data: {
                endpoints: [
                    { provider_name: 'Google' },
                    { provider_name: 'Anthropic' },
                    { provider_name: 'Google' },
                    { provider_name: null },
                ],
            },
        };

        expect(getOpenRouterProviderNames(response, true)).toEqual(['Anthropic', 'Google']);
    });

    test('extracts all providers when no concrete model is selected', () => {
        const response = {
            data: [
                { name: 'Together' },
                { name: 'OpenAI' },
                { name: '' },
            ],
        };

        expect(getOpenRouterProviderNames(response, false)).toEqual(['OpenAI', 'Together']);
    });

    test('handles an unexpected response shape', () => {
        expect(getOpenRouterProviderNames({}, true)).toEqual([]);
    });
});
