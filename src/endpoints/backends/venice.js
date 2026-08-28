import { createHmac } from 'node:crypto';

const VENICE_PROMPT_CACHE_CONTEXT = 'SillyTavern Venice prompt cache key v1';

/**
 * Normalizes Venice's live model response for the Chat Completion model picker.
 * @param {any} payload Venice models response
 * @param {Record<string, string>} [traits] Live Venice trait mappings
 * @returns {Record<string, any>[]} Available text models and the default trait
 */
export function normalizeVeniceModels(payload, traits = {}) {
    if (!Array.isArray(payload?.data)) {
        return [];
    }

    const models = payload.data
        .filter(model => model?.id && model?.type === 'text' && model?.model_spec?.offline !== true)
        .map(model => ({
            ...model,
            context_length: model.context_length ?? model.model_spec?.availableContextTokens,
            max_completion_tokens: model.model_spec?.maxCompletionTokens,
        }));

    const defaultModel = models.find(model => model.id === traits.default)
        ?? models.find(model => Array.isArray(model.model_spec?.traits) && model.model_spec.traits.includes('default'));
    if (!defaultModel) {
        return models;
    }

    const defaultTrait = { ...defaultModel, id: 'default', canonical_id: defaultModel.id, model_spec: { ...defaultModel.model_spec, name: 'Venice Default' } };
    return [defaultTrait, ...models];
}

/**
 * Builds the Venice-specific request extension from frontend settings.
 * @param {Record<string, any>} body Frontend generation request
 * @returns {Record<string, any>} Venice request extension
 */
export function buildVeniceParameters(body) {
    const searchMode = ['auto', 'off', 'on'].includes(body.venice_web_search)
        ? body.venice_web_search
        : 'off';

    return {
        character_slug: String(body.venice_character_slug || '').trim() || undefined,
        strip_thinking_response: Boolean(body.venice_strip_thinking_response),
        disable_thinking: Boolean(body.venice_disable_thinking),
        enable_web_search: searchMode,
        enable_web_scraping: Boolean(body.venice_enable_web_scraping),
        enable_web_citations: Boolean(body.venice_enable_web_citations),
        include_venice_system_prompt: Boolean(body.venice_include_system_prompt),
        enable_x_search: Boolean(body.venice_enable_x_search),
    };
}

/**
 * Derives an opaque, stable prompt-cache key scoped to a local user and chat.
 * @param {string|Buffer} serverSecret Server cookie secret
 * @param {string} userHandle Local user handle
 * @param {string} chatId Local chat identifier
 * @returns {string} 32-character opaque cache key
 */
export function deriveVenicePromptCacheKey(serverSecret, userHandle, chatId) {
    const purposeKey = createHmac('sha256', serverSecret).update(VENICE_PROMPT_CACHE_CONTEXT).digest();
    return createHmac('sha256', purposeKey)
        .update(userHandle)
        .update('\0')
        .update(chatId)
        .digest('hex')
        .slice(0, 32);
}
