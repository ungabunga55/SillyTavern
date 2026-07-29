import { createHmac } from 'node:crypto';

import { getCookieSecret } from '../../users.js';

const OPENROUTER_SESSION_HMAC_CONTEXT = 'SillyTavern OpenRouter session id v1';

/**
 * Builds an opaque OpenRouter session id from a local user and chat.
 * @param {string} secret Server cookie secret
 * @param {string} userHandle Local user handle
 * @param {string} chatId Local chat identifier
 * @returns {string} OpenRouter session id
 */
export function deriveOpenRouterSessionId(secret, userHandle, chatId) {
    const hmacKey = createHmac('sha256', secret)
        .update(OPENROUTER_SESSION_HMAC_CONTEXT)
        .digest();

    return createHmac('sha256', hmacKey)
        .update(userHandle)
        .update('\0')
        .update(chatId)
        .digest('hex')
        .slice(0, 32);
}

/**
 * Gets a user-scoped OpenRouter session id for the current local chat.
 * @param {import('express').Request} request Express request
 * @returns {string|undefined} OpenRouter session id
 */
export function getOpenRouterSessionId(request) {
    if (!request.body.openrouter_sticky_routing || typeof request.body.chat_id !== 'string') {
        return undefined;
    }

    const chatId = request.body.chat_id.slice(0, 512);
    if (!chatId) {
        return undefined;
    }

    const userHandle = typeof request.user?.profile?.handle === 'string' ? request.user.profile.handle : '';
    return deriveOpenRouterSessionId(getCookieSecret(globalThis.DATA_ROOT), userHandle, chatId);
}
