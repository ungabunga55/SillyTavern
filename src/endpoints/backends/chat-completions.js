/* eslint-disable dot-notation */
import process from 'node:process';
import util from 'node:util';
import { createHmac } from 'node:crypto';
import { Transform } from 'node:stream';
import express from 'express';
import fetch from 'node-fetch';
import urlJoin from 'url-join';

import {
    AGENTROUTER_HEADERS,
    AIMLAPI_HEADERS,
    AZURE_OPENAI_KEYS,
    CHAT_COMPLETION_SOURCES,
    FEATHERLESS_HEADERS,
    GEMINI_SAFETY,
    NANOGPT_REASONING_EFFORT_MAP,
    OPENAI_FIXED_REASONING_EFFORT,
    OPENAI_REASONING_EFFORT_MAP,
    OPENAI_REASONING_EFFORT_MODELS,
    OPENAI_VERBOSITY_MODELS,
    getOpenRouterHeaders,
    REQUESTY_HEADERS,
    VERTEX_SAFETY,
    SILICONFLOW_ENDPOINT,
    MINIMAX_ENDPOINT,
    ZAI_ENDPOINT,
    POLLINATIONS_ENDPOINT,
} from '../../constants.js';
import {
    forwardFetchResponse,
    getConfigValue,
    tryParse,
    uuidv4,
    mergeObjectWithYaml,
    excludeKeysByYaml,
    color,
    trimTrailingSlash,
    flattenSchema,
} from '../../util.js';
import {
    convertClaudeMessages,
    convertGooglePrompt,
    convertTextCompletionPrompt,
    convertCohereMessages,
    convertMistralMessages,
    convertAI21Messages,
    convertXAIMessages,
    cachingAtDepthForOpenRouterClaude,
    cachingAtDepthForClaude,
    getPromptNames,
    calculateClaudeBudgetTokens,
    calculateGoogleBudgetTokens,
    isGeminiNoSamplingModel,
    postProcessPrompt,
    PROMPT_PROCESSING_TYPE,
    addAssistantPrefix,
    extractMoonshotThinkingPrefill,
    embedOpenRouterMedia,
    addReasoningContentToToolCalls,
    cachingSystemPromptForOpenRouter,
    addOpenRouterSignatures,
    supportsClaudeMidConversationSystemMessages,
    shouldDisableClaudeThinking,
} from '../../prompt-converters.js';

import { readSecret, SECRET_KEYS } from '../secrets.js';
import {
    getTokenizerModel,
    getSentencepiceTokenizer,
    getTiktokenTokenizer,
    sentencepieceTokenizers,
    TEXT_COMPLETION_MODELS,
    webTokenizers,
    getWebTokenizer,
} from '../tokenizers.js';
import { getVertexAIAuth, getProjectIdFromServiceAccount } from '../google.js';
import { getCookieSecret } from '../../users.js';
import { getOpenRouterSessionId } from './openrouter-cache.js';

const API_OPENAI = 'https://api.openai.com/v1';
const API_AGENTROUTER = 'https://agentrouter.org/v1';
const API_CLAUDE = 'https://api.anthropic.com/v1';
const API_MISTRAL = 'https://api.mistral.ai/v1';
const API_COHERE_V1 = 'https://api.cohere.ai/v1';
const API_COHERE_V2 = 'https://api.cohere.ai/v2';
const API_PERPLEXITY = 'https://api.perplexity.ai';
const API_GROQ = 'https://api.groq.com/openai/v1';
const API_NVIDIA = 'https://integrate.api.nvidia.com/v1';
const API_MAKERSUITE = 'https://generativelanguage.googleapis.com';
const API_VERTEX_AI = 'https://us-central1-aiplatform.googleapis.com';
const API_AI21 = 'https://api.ai21.com/studio/v1';
const API_CHUTES = 'https://llm.chutes.ai/v1';
const API_ELECTRONHUB = 'https://api.electronhub.ai/v1';
const API_FEATHERLESS = 'https://api.featherless.ai/v1';
const API_NANOGPT = 'https://nano-gpt.com/api/v1';
const API_DEEPSEEK = 'https://api.deepseek.com/beta';
const API_XAI = 'https://api.x.ai/v1';
const API_META = 'https://api.meta.ai/v1';
const API_AIMLAPI = 'https://api.aimlapi.com/v1';
const API_POLLINATIONS = 'https://gen.pollinations.ai/v1';
const API_POLLINATIONS_ANON = 'https://text.pollinations.ai/v1';
const API_MOONSHOT = 'https://api.moonshot.ai/v1';
const API_FIREWORKS = 'https://api.fireworks.ai/inference/v1';
const API_FIREWORKS_MODELS = 'https://api.fireworks.ai/v1';
const FIREWORKS_DEFAULT_MODEL = 'accounts/fireworks/models/glm-5p2';
const FIREWORKS_LEGACY_DEFAULT_MODEL = 'accounts/fireworks/models/kimi-k2-instruct';
const OPENAI_PROMPT_CACHE_HMAC_CONTEXT = 'SillyTavern OpenAI prompt cache key v1';
const FIREWORKS_PROMPT_CACHE_HMAC_CONTEXT = 'SillyTavern Fireworks prompt cache affinity v1';
const XAI_PROMPT_CACHE_HMAC_CONTEXT = 'SillyTavern xAI prompt cache key v1';
const FIREWORKS_SERVERLESS_MODELS = [
    { id: 'accounts/fireworks/models/glm-5p3', display_name: 'GLM 5.3', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 1048576 },
    { id: 'accounts/fireworks/models/glm-5p2', display_name: 'GLM 5.2', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 1048576 },
    { id: 'accounts/fireworks/models/deepseek-v4-pro', display_name: 'DeepSeek-V4-Pro', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 1048576 },
    { id: 'accounts/fireworks/models/minimax-m2p7', display_name: 'MiniMax M2.7', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 196608 },
    { id: 'accounts/fireworks/models/glm-5p1', display_name: 'GLM 5.1', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 202752 },
    { id: 'accounts/fireworks/models/deepseek-v4-flash', display_name: 'DeepSeek-V4-Flash', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 1048576 },
    { id: 'accounts/fireworks/models/kimi-k2p7-code', display_name: 'Kimi K2.7 Code', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true, context_length: 262144 },
    { id: 'accounts/fireworks/models/kimi-k2p6', display_name: 'Kimi K2.6', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true, context_length: 262144 },
    { id: 'accounts/fireworks/models/qwen3p7-plus', display_name: 'Qwen3.7 Plus', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true },
    { id: 'accounts/fireworks/models/qwen3p6-plus', display_name: 'Qwen3.6 Plus', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true },
    { id: 'accounts/fireworks/models/nemotron-3-ultra-nvfp4', display_name: 'NVIDIA Nemotron 3 Ultra NVFP4', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 262144 },
    { id: 'accounts/fireworks/models/minimax-m3', display_name: 'MiniMax M3', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true, context_length: 512000 },
    { id: 'accounts/fireworks/models/gpt-oss-120b', display_name: 'OpenAI gpt-oss-120b', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 131072 },
    { id: 'accounts/fireworks/models/gpt-oss-20b', display_name: 'OpenAI gpt-oss-20b', supports_chat: true, supports_tools: false, supports_image_in: false, supports_serverless: true, context_length: 131072 },
    { id: 'accounts/fireworks/routers/kimi-k2p7-code-fast', display_name: 'Kimi K2.7 Code Fast', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true, context_length: 262144 },
    { id: 'accounts/fireworks/routers/kimi-k2p6-fast', display_name: 'Kimi K2.6 Fast', supports_chat: true, supports_tools: true, supports_image_in: true, supports_serverless: true, context_length: 262144 },
    { id: 'accounts/fireworks/routers/glm-5p3-fast', display_name: 'GLM 5.3 Fast', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 1048576 },
    { id: 'accounts/fireworks/routers/glm-5p2-fast', display_name: 'GLM 5.2 Fast', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 1048576 },
    { id: 'accounts/fireworks/routers/glm-5p1-fast', display_name: 'GLM 5.1 Fast', supports_chat: true, supports_tools: true, supports_image_in: false, supports_serverless: true, context_length: 202752 },
];
const API_COMETAPI = 'https://api.cometapi.com/v1';
const API_ZAI_COMMON = 'https://api.z.ai/api/paas/v4';
const API_ZAI_CODING = 'https://api.z.ai/api/coding/paas/v4';
const API_SILICONFLOW = 'https://api.siliconflow.com/v1';
const API_SILICONFLOW_CN = 'https://api.siliconflow.cn/v1';
const API_ATLASCLOUD = 'https://api.atlascloud.ai/v1';
const API_MINIMAX = 'https://api.minimax.io/v1';
const API_MINIMAX_CN = 'https://api.minimaxi.com/v1';
const API_OPENROUTER = 'https://openrouter.ai/api/v1';
const API_REQUESTY = 'https://router.requesty.ai/v1';
const API_WORKERS_AI = 'https://api.cloudflare.com/client/v4/accounts';

const NVIDIA_DEFAULT_ENABLED_PARAMETERS = [
    'temperature',
    'top_p',
    'frequency_penalty',
    'presence_penalty',
    'top_k',
    'repetition_penalty',
    'min_p',
    'top_a',
    'seed',
    'thinking',
    'reasoning_effort',
];

const MOONSHOT_KIMI_FIXED_PARAMETER_MODEL_REGEX = /^kimi-k2(?:\.5|\.6|\.7-code|-0905-preview|-turbo-preview|-thinking|-thinking-turbo)$/;
const MOONSHOT_KIMI_K3_MODEL_REGEX = /^kimi-k3(?:$|[-.])/;
const XAI_REASONING_EFFORTS = new Set(['none', 'low', 'medium', 'high', 'xhigh']);
let openaiPromptCacheHmacKey;
let fireworksPromptCacheHmacKey;
let xaiPromptCacheHmacKey;
const CLAUDE_LEGACY_SAMPLING_MODEL_REGEXES = [
    /^claude-2(?:$|[-.])/,
    /^claude-instant(?:$|-)/,
    /^claude-3(?:$|-)/,
    /^claude-(?:opus|sonnet)-4(?:$|-20\d{6})/,
    /^claude-opus-4-(?:1|5|6)(?:$|-)/,
    /^claude-sonnet-4-(?:5|6)(?:$|-)/,
    /^claude-haiku-4-5(?:$|-)/,
];
const CLAUDE_LIMITED_SAMPLING_MODEL_REGEXES = [
    /^claude-opus-4-(?:1|5|6)(?:$|-)/,
    /^claude-sonnet-4-(?:5|6)(?:$|-)/,
    /^claude-haiku-4-5(?:$|-)/,
];

/**
 * Checks if a Moonshot Kimi model only accepts fixed sampler values.
 * @param {string} model Model identifier
 * @returns {boolean} True if the model rejects modified sampler values
 */
function isMoonshotKimiFixedParameterModel(model) {
    return MOONSHOT_KIMI_FIXED_PARAMETER_MODEL_REGEX.test(String(model || ''));
}

/**
 * Checks if a Moonshot model belongs to the Kimi K3 family.
 * @param {string} model Model identifier
 * @returns {boolean} True if the model uses K3 reasoning effort controls
 */
function isMoonshotKimiK3Model(model) {
    return MOONSHOT_KIMI_K3_MODEL_REGEX.test(String(model || ''));
}

/**
 * Checks if an OpenRouter model belongs to the Moonshot provider family.
 * @param {string} model Model identifier
 * @returns {boolean} True if the model is published by Moonshot
 */
function isOpenRouterMoonshotModel(model) {
    return /(?:^|\/)moonshotai\//i.test(String(model || ''));
}

/**
 * Checks if a Featherless model is published by Moonshot.
 * @param {string} model Model identifier
 * @returns {boolean} True if the model belongs to Moonshot
 */
function isFeatherlessMoonshotModel(model) {
    return /(?:^|\/)moonshotai\//i.test(String(model || ''));
}

/**
 * Checks if a Moonshot Kimi model always has thinking enabled.
 * @param {string} model Model identifier
 * @returns {boolean} True if thinking cannot be disabled
 */
function isMoonshotKimiAlwaysOnThinkingModel(model) {
    return /^kimi-k2\.7-code$/.test(String(model || ''));
}

/**
 * Normalizes a Claude model id for capability checks.
 * @param {string} model Model identifier
 * @returns {string} Normalized model id
 */
function getClaudeModelId(model) {
    return String(model || '').toLowerCase().trim();
}

/**
 * Checks if a model id identifies an Anthropic Claude model.
 * @param {string} model Model identifier
 * @returns {boolean} True if the model is Claude
 */
function isClaudeModel(model) {
    const modelId = getClaudeModelId(model);
    return modelId.startsWith('claude-') || modelId.startsWith('claude.');
}

/**
 * Checks if a Claude model is known to still accept sampling parameters.
 * Unknown future Claude models default to no-sampling because newer Claude
 * releases reject temperature, top_p, and top_k entirely.
 * @param {string} model Model identifier
 * @returns {boolean} True if sampling parameters may be sent
 */
function isClaudeLegacySamplingModel(model) {
    const modelId = getClaudeModelId(model);
    return CLAUDE_LEGACY_SAMPLING_MODEL_REGEXES.some(regex => regex.test(modelId));
}

/**
 * Checks if a Claude model applies the temperature/top_p mutual-exclusion rule.
 * @param {string} model Model identifier
 * @returns {boolean} True if only one sampler should be sent
 */
function isClaudeLimitedSamplingModel(model) {
    const modelId = getClaudeModelId(model);
    return CLAUDE_LIMITED_SAMPLING_MODEL_REGEXES.some(regex => regex.test(modelId));
}

/**
 * Checks if a Claude model rejects sampling parameters entirely.
 * @param {string} model Model identifier
 * @returns {boolean} True if temperature, top_p, and top_k should be stripped
 */
function isClaudeNoSamplingModel(model) {
    return isClaudeModel(model) && !isClaudeLegacySamplingModel(model);
}

/**
 * Checks if a Claude model supports thinking mode.
 * @param {string} model Model identifier
 * @returns {boolean} True if thinking may be sent
 */
function isClaudeThinkingModel(model) {
    const modelId = getClaudeModelId(model);
    return /^claude-3-7(?:$|-)/.test(modelId)
        || /^claude-(?:opus|sonnet)-4(?:$|-20\d{6})/.test(modelId)
        || isClaudeLimitedSamplingModel(modelId)
        || isClaudeNoSamplingModel(modelId);
}

/**
 * Checks if a Claude model uses adaptive thinking effort.
 * @param {string} model Model identifier
 * @param {boolean} enableLegacyAdaptive Whether to enable adaptive mode for transitional 4.6 models
 * @returns {boolean} True if adaptive thinking should be used
 */
function isClaudeAdaptiveThinkingModel(model, enableLegacyAdaptive = true) {
    const modelId = getClaudeModelId(model);
    return isClaudeNoSamplingModel(modelId)
        || (enableLegacyAdaptive && /^claude-(?:opus-4-6|sonnet-4-6)(?:$|-)/.test(modelId));
}

/**
 * Checks if a Claude model rejects assistant prefill in thinking/adaptive modes.
 * @param {string} model Model identifier
 * @returns {boolean} True if assistant prefill should be converted
 */
function isClaudeNoPrefillModel(model) {
    const modelId = getClaudeModelId(model);
    return isClaudeNoSamplingModel(modelId) || /^claude-(?:opus-4-6|sonnet-4-6)(?:$|-)/.test(modelId);
}

/**
 * Checks if a Claude model supports verbosity/effort output config.
 * @param {string} model Model identifier
 * @returns {boolean} True if verbosity may be sent
 */
function isClaudeVerbosityModel(model) {
    const modelId = getClaudeModelId(model);
    return isClaudeNoSamplingModel(modelId) || /^claude-(?:opus-4-5|opus-4-6|sonnet-4-6)(?:$|-)/.test(modelId);
}

/**
 * Checks if adaptive thinking should be emitted even with automatic effort.
 * @param {string} model Model identifier
 * @returns {boolean} True if adaptive thinking is forced on
 */
function isClaudeForcedAdaptiveThinkingModel(model) {
    return /^claude-(?:fable-5|mythos-5|mythos-preview)(?:$|-)/.test(getClaudeModelId(model));
}

/**
 * Gets the OpenAI Responses-compatible reasoning effort for a model.
 * @param {string} model Model identifier
 * @param {string} effort User-selected effort
 * @returns {string|undefined} Mapped effort, if supported
 */
function getOpenAIReasoningEffort(model, effort) {
    if (!effort || !OPENAI_REASONING_EFFORT_MODELS.includes(model)) {
        return undefined;
    }

    if (/^gpt-5\.6/.test(String(model || '').toLowerCase())) {
        if (effort === 'min') {
            return 'none';
        }
        if (['none', 'low', 'medium', 'high', 'xhigh', 'max'].includes(effort)) {
            return effort;
        }
    }

    return OPENAI_FIXED_REASONING_EFFORT[model] ?? OPENAI_REASONING_EFFORT_MAP[effort] ?? effort;
}

/**
 * Checks if an OpenAI model supports Responses reasoning.mode.
 * @param {string} model Model identifier
 * @returns {boolean} True if reasoning.mode may be sent
 */
function isOpenAIReasoningModeModel(model) {
    const modelId = String(model || '').toLowerCase();
    return /^gpt-5\.(?:[6-9]|\d{2,})/.test(modelId) || /^gpt-(?:[6-9]|\d{2,})/.test(modelId);
}

/**
 * Gets the OpenAI Responses-compatible reasoning mode for a model.
 * @param {string} model Model identifier
 * @param {string} mode User-selected mode
 * @returns {string|undefined} Mode, if supported
 */
function getOpenAIReasoningMode(model, mode) {
    if (!isOpenAIReasoningModeModel(model)) {
        return undefined;
    }

    return ['standard', 'pro'].includes(mode) ? mode : undefined;
}

/**
 * Checks if a Responses request should omit sampling parameters.
 * @param {string} model Model identifier
 * @param {string|undefined} effort Reasoning effort
 * @returns {boolean} True if temperature/top_p should not be sent
 */
function isOpenAIResponsesNoSamplingModel(model, effort) {
    const modelId = String(model || '').toLowerCase();
    if (/^(o1|o3|o4)/.test(modelId)) {
        return true;
    }
    if (/^gpt-5\.[56]/.test(modelId)) {
        return true;
    }
    return modelId.startsWith('gpt-5') && Boolean(effort) && effort !== 'none';
}

/**
 * Removes undefined values from an object.
 * @param {Record<string, any>} object Object to clean
 * @returns {Record<string, any>} Clean object
 */
function removeUndefinedValues(object) {
    return Object.fromEntries(Object.entries(object).filter(([, value]) => value !== undefined));
}

/**
 * Applies Atlascloud's stricter OpenAI-compatible request schema and mirrors
 * model-family rules used by the native provider integrations where possible.
 * @param {Record<string, any>} requestBody Request payload to mutate
 * @param {express.Request} request Express request
 */
function sanitizeAtlascloudRequestBody(requestBody, request) {
    const model = String(request.body.model || '');
    const modelId = model.toLowerCase();
    const nativeModel = modelId.includes('/') ? modelId.split('/').slice(1).join('/') : modelId;
    const family = getAtlascloudModelFamily(modelId);
    const includeReasoning = Boolean(request.body.include_reasoning);
    const thinkingPrefill = family === 'moonshot' && Boolean(request.body.moonshot_thinking_prefill) && !request.body.json_schema;
    const preservedThinking = family === 'moonshot' && Boolean(request.body.moonshot_preserved_thinking);
    const thinkingEnabled = family === 'moonshot'
        ? isMoonshotKimiAlwaysOnThinkingModel(nativeModel) || includeReasoning || thinkingPrefill || preservedThinking
        : includeReasoning;

    function getReasoningEffort() {
        const effort = String(request.body.reasoning_effort || 'auto');
        if (!includeReasoning || effort === 'auto') {
            return undefined;
        }

        switch (family) {
            case 'openai':
                if (!OPENAI_REASONING_EFFORT_MODELS.includes(nativeModel)) {
                    return undefined;
                }
                return getOpenAIReasoningEffort(nativeModel, effort);
            case 'xai':
                if (effort === 'min') {
                    return 'none';
                }
                if (['xhigh', 'max'].includes(effort)) {
                    return 'high';
                }
                return effort;
            case 'deepseek':
                return effort === 'max' ? 'max' : 'high';
            case 'zai':
                if (!/(?:^|\/)glm-5\.[23](?:$|\b)/.test(modelId)) {
                    return undefined;
                }
                return effort === 'min' ? 'minimal' : effort;
            default:
                return undefined;
        }
    }

    delete requestBody.prompt;
    delete requestBody.max_completion_tokens;
    delete requestBody.presence_penalty;
    delete requestBody.frequency_penalty;
    delete requestBody.stop;
    delete requestBody.logit_bias;
    delete requestBody.seed;
    delete requestBody.n;
    delete requestBody.response_format;
    delete requestBody.tools;
    delete requestBody.tool_choice;
    delete requestBody.thinking;
    delete requestBody.reasoning_effort;

    if (!Number.isFinite(Number(requestBody.top_k)) || Number(requestBody.top_k) <= 0) {
        delete requestBody.top_k;
    }

    if (!Number.isFinite(Number(requestBody.repetition_penalty)) || Number(requestBody.repetition_penalty) === 1) {
        delete requestBody.repetition_penalty;
    }

    if (!thinkingEnabled) {
        requestBody.thinking = { type: 'disabled' };
    } else {
        requestBody.thinking = { type: 'enabled' };

        const reasoningEffort = getReasoningEffort();
        if (reasoningEffort) {
            requestBody.reasoning_effort = reasoningEffort;
        }
    }

    if (family === 'zai') {
        requestBody.top_p = requestBody.top_p || 0.01;
        delete requestBody.top_k;
        delete requestBody.repetition_penalty;
    }

    if (family === 'deepseek') {
        requestBody.top_p = requestBody.top_p || Number.EPSILON;
    }

    if (family === 'xai') {
        requestBody.max_completion_tokens = request.body.max_completion_tokens ?? requestBody.max_tokens;
        delete requestBody.max_tokens;
        requestBody.top_p = requestBody.top_p || Number.EPSILON;
    }

    if (family === 'google') {
        delete requestBody.repetition_penalty;
    }

    if (['openai', 'xai', 'deepseek', 'moonshot', 'minimax'].includes(family)) {
        delete requestBody.top_k;
        delete requestBody.repetition_penalty;
    }

    if (family === 'openai' && /^(o1|o3|o4)/.test(nativeModel)) {
        requestBody.max_completion_tokens = request.body.max_completion_tokens ?? requestBody.max_tokens;
        delete requestBody.max_tokens;
        delete requestBody.temperature;
        delete requestBody.top_p;
        if (/^o1/.test(nativeModel) && Array.isArray(requestBody.messages)) {
            requestBody.messages.forEach(message => {
                if (message?.role === 'system') {
                    message.role = 'user';
                }
            });
        }
    }

    if (family === 'openai' && /gpt-5/.test(nativeModel)) {
        requestBody.max_completion_tokens = request.body.max_completion_tokens ?? requestBody.max_tokens;
        delete requestBody.max_tokens;

        if (/gpt-5-chat-latest/.test(nativeModel)) {
            // no-op: chat-latest keeps sampling parameters in the direct OpenAI path.
        } else if (/gpt-5\.(1|2|3|4)/.test(nativeModel) && !/chat-latest/.test(nativeModel) && !requestBody.reasoning_effort) {
            // Keep temperature/top_p, matching direct OpenAI chat-completions handling.
        } else {
            delete requestBody.temperature;
            delete requestBody.top_p;
        }
    }

    if (family === 'moonshot' && isMoonshotKimiFixedParameterModel(nativeModel)) {
        delete requestBody.temperature;
        delete requestBody.top_p;
    }

    if (family === 'moonshot' && isMoonshotKimiAlwaysOnThinkingModel(nativeModel)) {
        requestBody.thinking = { type: 'enabled' };
    }

    if (family === 'moonshot') {
        normalizeMoonshotReasoningContent(requestBody.messages, thinkingEnabled, preservedThinking);
        if (thinkingPrefill) {
            extractMoonshotThinkingPrefill(requestBody.messages);
            addAssistantPrefix(requestBody.messages, [], 'partial', true);
        }
    }

    if (family === 'claude') {
        const useThinking = isClaudeThinkingModel(nativeModel);
        const isLimitedSampling = isClaudeLimitedSamplingModel(nativeModel);
        const isAdaptiveModel = isClaudeAdaptiveThinkingModel(nativeModel, enableAdaptiveThinking);
        const noSamplingModel = isClaudeNoSamplingModel(nativeModel);

        delete requestBody.repetition_penalty;

        if (isLimitedSampling) {
            if (requestBody.top_p < 1) {
                delete requestBody.temperature;
            } else {
                delete requestBody.top_p;
            }
        }

        if (noSamplingModel) {
            delete requestBody.temperature;
            delete requestBody.top_p;
            delete requestBody.top_k;
        }

        if (includeReasoning && isAdaptiveModel) {
            requestBody.thinking = { type: 'adaptive', display: 'summarized' };
            delete requestBody.top_k;
        } else if (includeReasoning && useThinking) {
            if (Number(requestBody.max_tokens) <= 1024) {
                requestBody.max_tokens = Number(requestBody.max_tokens) + 1024;
            }
            delete requestBody.temperature;
            delete requestBody.top_p;
            delete requestBody.top_k;
        }
    }

    if (family === 'minimax') {
        requestBody.messages = postProcessPrompt(requestBody.messages, PROMPT_PROCESSING_TYPE.MERGE_TOOLS, getPromptNames(request));
    }

    if (family === 'minimax' && Number.isFinite(Number(requestBody.temperature))) {
        requestBody.temperature = Math.min(Math.max(Number(requestBody.temperature), Number.EPSILON), 1.0);
    }
}

/**
 * Gets the closest direct provider family for an Atlascloud model id.
 * @param {string} modelId Lower-cased Atlascloud model id
 * @returns {string} Provider family name
 */
function getAtlascloudModelFamily(modelId) {
    if (modelId.startsWith('openai/')) return 'openai';
    if (modelId.startsWith('anthropic/claude-') || modelId.startsWith('anthropic/claude.')) return 'claude';
    if (modelId.startsWith('google/gemini-')) return 'google';
    if (modelId.startsWith('xai/')) return 'xai';
    if (modelId.startsWith('zai-org/') || /(?:^|\/)glm-/.test(modelId)) return 'zai';
    if (modelId.startsWith('deepseek-ai/') || modelId.includes('deepseek')) return 'deepseek';
    if (modelId.startsWith('moonshotai/') || modelId.includes('kimi')) return 'moonshot';
    if (modelId.startsWith('minimaxai/') || modelId.includes('minimax')) return 'minimax';
    if (modelId.startsWith('qwen/')) return 'qwen';
    if (modelId.startsWith('bytedance/')) return 'doubao';
    return 'generic';
}

/**
 * Normalizes a Fireworks model record to the frontend's OpenAI-compatible metadata shape.
 * @param {Record<string, any>} model Fireworks model record or static model record
 * @returns {Record<string, any>} Normalized model metadata
 */
function normalizeFireworksModel(model) {
    const id = String(model?.id || model?.name || '').trim();
    const isEmbeddingModel = model?.kind === 'EMBEDDING_MODEL' || /(?:embedding|reranker)/i.test(id);
    const supportsChat = Boolean(model?.supports_chat ?? (model?.conversationConfig && !isEmbeddingModel));
    const contextLength = model?.context_length ?? model?.contextLength;
    const displayName = model?.display_name ?? model?.displayName;
    const supportsTools = model?.supports_tools ?? model?.supportsTools;
    const supportsImageInput = model?.supports_image_in ?? model?.supportsImageInput;
    const supportsServerless = model?.supports_serverless ?? model?.supportsServerless;
    const deprecationDate = model?.deprecation_date ?? model?.deprecationDate;
    const defaultSamplingParams = model?.default_sampling_params ?? model?.defaultSamplingParams;

    return {
        id,
        display_name: displayName,
        supports_chat: supportsChat,
        supports_tools: Boolean(supportsTools),
        supports_image_in: Boolean(supportsImageInput),
        supports_serverless: Boolean(supportsServerless ?? true),
        context_length: Number.isFinite(Number(contextLength)) && Number(contextLength) > 0 ? Number(contextLength) : undefined,
        deprecation_date: deprecationDate,
        default_sampling_params: defaultSamplingParams,
    };
}

/**
 * Merges dynamic Fireworks model metadata with static fallbacks and router supplements.
 * @param {Record<string, any>[]} models Dynamic model records from Fireworks
 * @returns {Record<string, any>[]} Chat-capable model metadata records
 */
function mergeFireworksModels(models) {
    const merged = new Map();
    for (const model of [...models, ...FIREWORKS_SERVERLESS_MODELS]) {
        const normalized = normalizeFireworksModel(model);
        if (!normalized.id || !normalized.supports_chat) {
            continue;
        }
        const existing = merged.get(normalized.id) || {};
        merged.set(normalized.id, { ...normalized, ...existing });
    }
    return Array.from(merged.values());
}

/**
 * Fetches Fireworks serverless model metadata. Falls back only for non-auth endpoint failures.
 * @param {string} apiKey Fireworks API key
 * @returns {Promise<{ models: Record<string, any>[], fallback: boolean }>}
 */
async function fetchFireworksServerlessModels(apiKey) {
    const models = [];
    let pageToken = '';

    for (let page = 0; page < 10; page++) {
        const modelsUrl = new URL('/v1/accounts/fireworks/models', API_FIREWORKS_MODELS);
        modelsUrl.searchParams.set('filter', 'supports_serverless=true');
        modelsUrl.searchParams.set('pageSize', '200');
        if (pageToken) {
            modelsUrl.searchParams.set('pageToken', pageToken);
        }

        const response = await fetch(modelsUrl, {
            method: 'GET',
            headers: {
                'Authorization': 'Bearer ' + apiKey,
            },
        });

        if (!response.ok) {
            if ([401, 403].includes(response.status)) {
                const error = new Error(`Fireworks AI models endpoint failed: ${response.status} ${response.statusText}`);
                error['status'] = response.status;
                throw error;
            }
            console.warn('Fireworks AI models endpoint failed, using static fallback:', response.status, response.statusText);
            return { models: mergeFireworksModels([]), fallback: true };
        }

        /** @type {any} */
        const data = await response.json();
        if (Array.isArray(data?.models)) {
            models.push(...data.models);
        }
        pageToken = String(data?.nextPageToken || '');
        if (!pageToken) {
            break;
        }
    }

    return { models: mergeFireworksModels(models), fallback: false };
}

/**
 * Applies direct-provider-derived request rules for Fireworks serverless models.
 * @param {Record<string, any>} requestBody Request payload to mutate
 * @param {express.Request} request Express request
 */
function sanitizeFireworksRequestBody(requestBody, request) {
    const modelId = String(request.body.model || '').toLowerCase();
    const nativeModel = modelId.split('/').pop() || modelId;
    const family = getFireworksModelFamily(modelId);
    const includeReasoning = Boolean(request.body.include_reasoning);
    const isMoonshot = isFireworksMoonshotModel(modelId);
    const thinkingPrefill = isMoonshot && Boolean(request.body.moonshot_thinking_prefill) && !request.body.json_schema;
    const preservedThinking = isMoonshot && Boolean(request.body.moonshot_preserved_thinking);
    const useMoonshotOptions = thinkingPrefill || preservedThinking;

    function getReasoningEffort() {
        const effort = String(request.body.reasoning_effort || 'auto');
        if (!includeReasoning || effort === 'auto') {
            return undefined;
        }

        switch (family) {
            case 'zai':
                if (['glm-5p2', 'glm-5p3'].includes(nativeModel)) {
                    if (effort === 'min') return 'none';
                    if (effort === 'xhigh') return 'max';
                    return effort;
                }
                return effort === 'min' ? 'none' : undefined;
            case 'deepseek':
                if (effort === 'min') return 'none';
                if (['xhigh', 'max'].includes(effort)) return 'max';
                return ['low', 'medium', 'high'].includes(effort) ? effort : 'high';
            case 'qwen':
                if (effort === 'min') return 'none';
                if (['xhigh', 'max'].includes(effort)) return 'high';
                return ['low', 'medium', 'high'].includes(effort) ? effort : undefined;
            case 'minimax':
                if (nativeModel !== 'minimax-m2p7') return undefined;
                if (['low', 'medium', 'high'].includes(effort)) return effort;
                if (['xhigh', 'max'].includes(effort)) return 'high';
                return undefined;
            default:
                return undefined;
        }
    }

    delete requestBody.prompt;
    delete requestBody.max_completion_tokens;
    delete requestBody.reasoning_effort;

    if (Array.isArray(requestBody.stop) && requestBody.stop.length === 0) {
        delete requestBody.stop;
    }

    if (family === 'zai' || family === 'deepseek' || family === 'moonshot' || family === 'qwen' || useMoonshotOptions) {
        const thinkingEnabled = family === 'moonshot' || useMoonshotOptions
            ? isFireworksKimiAlwaysOnThinkingModel(nativeModel) || includeReasoning || thinkingPrefill || preservedThinking
            : includeReasoning;
        const reasoningEffort = getReasoningEffort();
        if (reasoningEffort) {
            requestBody.reasoning_effort = reasoningEffort;
        } else {
            requestBody.thinking = { type: thinkingEnabled ? 'enabled' : 'disabled' };
        }

        if (family === 'moonshot' || useMoonshotOptions) {
            normalizeMoonshotReasoningContent(requestBody.messages, thinkingEnabled, preservedThinking);
            if (thinkingPrefill) {
                extractMoonshotThinkingPrefill(requestBody.messages);
                addAssistantPrefix(requestBody.messages, [], 'partial', true);
            }
        }
    } else {
        delete requestBody.thinking;
    }

    if (family === 'zai') {
        requestBody.top_p = requestBody.top_p || 0.01;
        delete requestBody.frequency_penalty;
        delete requestBody.presence_penalty;
        if (Array.isArray(requestBody.stop) && requestBody.stop.length > 1) {
            requestBody.stop = requestBody.stop.slice(0, 1);
        }
    }

    if (family === 'deepseek') {
        requestBody.top_p = requestBody.top_p || Number.EPSILON;
        if (Array.isArray(requestBody.tools)) {
            requestBody.tools.forEach(tool => {
                const required = tool?.function?.parameters?.required;
                if (Array.isArray(required) && required.length === 0) {
                    delete tool.function.parameters.required;
                }
            });
        }
        if (request.body.json_schema && Array.isArray(requestBody.messages)) {
            requestBody.response_format = { type: 'json_object' };
            requestBody.messages.push({
                role: 'user',
                content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
            });
        }
    }

    if (family === 'moonshot') {
        delete requestBody.top_k;
        delete requestBody.repetition_penalty;

        if (isFireworksKimiFixedParameterModel(nativeModel)) {
            delete requestBody.temperature;
            delete requestBody.frequency_penalty;
            delete requestBody.presence_penalty;
            delete requestBody.top_p;
            delete requestBody.top_k;
            delete requestBody.repetition_penalty;
            delete requestBody.n;
        }

        if (Array.isArray(requestBody.tools)
            && requestBody.thinking?.type === 'enabled'
            && !['auto', 'none', undefined].includes(requestBody.tool_choice)) {
            requestBody.tool_choice = 'auto';
        }
    }

    if (family === 'minimax') {
        requestBody.messages = postProcessPrompt(requestBody.messages, PROMPT_PROCESSING_TYPE.MERGE_TOOLS, getPromptNames(request));
        const reasoningEffort = getReasoningEffort();
        if (reasoningEffort) {
            requestBody.reasoning_effort = reasoningEffort;
        }
        delete requestBody.frequency_penalty;
        delete requestBody.presence_penalty;
        delete requestBody.logit_bias;
        delete requestBody.seed;
        delete requestBody.n;
        if (Number.isFinite(Number(requestBody.temperature))) {
            requestBody.temperature = Math.min(Math.max(Number(requestBody.temperature), Number.EPSILON), 1.0);
        }
    }
}

/**
 * Gets a purpose-specific HMAC key for Fireworks prompt-cache affinity.
 * @returns {Buffer} Derived HMAC key
 */
function getFireworksPromptCacheHmacKey() {
    if (!fireworksPromptCacheHmacKey) {
        fireworksPromptCacheHmacKey = createHmac('sha256', getCookieSecret(globalThis.DATA_ROOT))
            .update(FIREWORKS_PROMPT_CACHE_HMAC_CONTEXT)
            .digest();
    }

    return fireworksPromptCacheHmacKey;
}

/**
 * Builds an opaque Fireworks session-affinity value from the current local chat.
 * @param {express.Request} request Express request
 * @returns {string|undefined} Session affinity value
 */
function getFireworksSessionAffinity(request) {
    if (!request.body.fireworks_prompt_caching || typeof request.body.chat_id !== 'string') {
        return undefined;
    }

    const chatId = request.body.chat_id.slice(0, 512);
    if (!chatId) {
        return undefined;
    }

    const userHandle = typeof request.user?.profile?.handle === 'string' ? request.user.profile.handle : '';
    return createHmac('sha256', getFireworksPromptCacheHmacKey())
        .update(userHandle)
        .update('\0')
        .update(chatId)
        .digest('hex')
        .slice(0, 32);
}

/**
 * Gets a purpose-specific HMAC key for OpenAI prompt caching.
 * @returns {Buffer} Derived HMAC key
 */
function getOpenAIPromptCacheHmacKey() {
    if (!openaiPromptCacheHmacKey) {
        openaiPromptCacheHmacKey = createHmac('sha256', getCookieSecret(globalThis.DATA_ROOT))
            .update(OPENAI_PROMPT_CACHE_HMAC_CONTEXT)
            .digest();
    }

    return openaiPromptCacheHmacKey;
}

/**
 * Builds an opaque OpenAI prompt-cache key from the current local chat.
 * @param {express.Request} request Express request
 * @returns {string|undefined} Prompt cache key
 */
function getOpenAIPromptCacheKey(request) {
    if (!request.body.openai_prompt_caching || typeof request.body.chat_id !== 'string') {
        return undefined;
    }

    const chatId = request.body.chat_id.slice(0, 512);
    if (!chatId) {
        return undefined;
    }

    const userHandle = typeof request.user?.profile?.handle === 'string' ? request.user.profile.handle : '';
    return createHmac('sha256', getOpenAIPromptCacheHmacKey())
        .update(userHandle)
        .update('\0')
        .update(chatId)
        .digest('hex')
        .slice(0, 32);
}

/**
 * Gets a purpose-specific HMAC key for xAI prompt caching.
 * @returns {Buffer} Derived HMAC key
 */
function getXaiPromptCacheHmacKey() {
    if (!xaiPromptCacheHmacKey) {
        xaiPromptCacheHmacKey = createHmac('sha256', getCookieSecret(globalThis.DATA_ROOT))
            .update(XAI_PROMPT_CACHE_HMAC_CONTEXT)
            .digest();
    }

    return xaiPromptCacheHmacKey;
}

/**
 * Builds an opaque xAI prompt-cache key from the current local chat.
 * @param {express.Request} request Express request
 * @returns {string|undefined} Prompt cache key
 */
function getXaiPromptCacheKey(request) {
    if (!request.body.xai_prompt_caching || typeof request.body.chat_id !== 'string') {
        return undefined;
    }

    const chatId = request.body.chat_id.slice(0, 512);
    if (!chatId) {
        return undefined;
    }

    const userHandle = typeof request.user?.profile?.handle === 'string' ? request.user.profile.handle : '';
    return createHmac('sha256', getXaiPromptCacheHmacKey())
        .update(userHandle)
        .update('\0')
        .update(chatId)
        .digest('hex')
        .slice(0, 32);
}

/**
 * Gets the closest direct provider family for a Fireworks model id.
 * @param {string} modelId Lower-cased Fireworks model id
 * @returns {string} Provider family name
 */
function getFireworksModelFamily(modelId) {
    const nativeModel = modelId.split('/').pop() || modelId;
    if (/^glm-5p[123]$/.test(nativeModel)) return 'zai';
    if (/^deepseek-v4-(pro|flash)$/.test(nativeModel)) return 'deepseek';
    if (nativeModel === 'kimi-k2p7-code') return 'moonshot';
    if (/^qwen3p\d-(?:plus|coder)$/.test(nativeModel)) return 'qwen';
    if (/^minimax-(m2p7|m3)$/.test(nativeModel)) return 'minimax';
    return 'generic';
}

/**
 * Checks if a Fireworks model belongs to the Moonshot family.
 * @param {string} modelId Lower-cased Fireworks model id
 * @returns {boolean} True if the model is a Moonshot model or router
 */
function isFireworksMoonshotModel(modelId) {
    const nativeModel = modelId.split('/').pop() || modelId;
    return /^kimi(?:$|[-_.])/.test(nativeModel) || /(?:^|\/)moonshotai(?:\/|$)/.test(modelId);
}

/**
 * Checks if a Fireworks Kimi model maps to a Moonshot fixed-parameter model.
 * @param {string} nativeModel Fireworks native model id
 * @returns {boolean} True if fixed sampler values are required
 */
function isFireworksKimiFixedParameterModel(nativeModel) {
    return nativeModel === 'kimi-k2p7-code';
}

/**
 * Checks if a Fireworks Kimi model maps to a Moonshot always-on thinking model.
 * @param {string} nativeModel Fireworks native model id
 * @returns {boolean} True if thinking should not be disabled
 */
function isFireworksKimiAlwaysOnThinkingModel(nativeModel) {
    return nativeModel === 'kimi-k2p7-code';
}

/**
 * Converts OpenAI Chat Completions message content to Responses input content.
 * @param {string|any[]} content Message content
 * @returns {string|any[]} Responses-compatible content
 */
function convertOpenAIResponsesContent(content) {
    if (!Array.isArray(content)) {
        return content ?? '';
    }

    return content.map(part => {
        if (part?.type === 'text') {
            return { type: 'input_text', text: part.text ?? '' };
        }

        if (part?.type === 'image_url') {
            return removeUndefinedValues({
                type: 'input_image',
                image_url: part.image_url?.url ?? part.image_url,
                detail: part.image_url?.detail,
            });
        }

        if (part?.type === 'file') {
            return removeUndefinedValues({
                type: 'input_file',
                filename: part.file?.filename ?? part.filename,
                file_data: part.file?.file_data ?? part.file_data,
                file_id: part.file?.file_id ?? part.file_id,
                file_url: part.file?.file_url ?? part.file_url,
            });
        }

        if (part?.type === 'video_url') {
            return removeUndefinedValues({
                type: 'input_video',
                video_url: part.video_url?.url ?? part.video_url,
            });
        }

        if (part?.type === 'input_audio') {
            return part;
        }

        return part;
    });
}

/**
 * Converts OpenAI Chat Completions messages to native Responses input items.
 * @param {string|object[]} messages Chat Completions messages or a text prompt
 * @returns {{instructions: string|undefined, input: object[]}} Responses input
 */
function convertOpenAIResponsesInput(messages) {
    if (!Array.isArray(messages)) {
        return { instructions: undefined, input: [{ type: 'message', role: 'user', content: String(messages ?? '') }] };
    }

    let instructions;
    const input = [];
    let sawNonSystemInput = false;

    for (const message of messages) {
        if (!message || typeof message !== 'object') {
            continue;
        }

        if (message.role === 'system' && typeof message.content === 'string' && message.content.trim()) {
            if (!sawNonSystemInput) {
                instructions = instructions ? `${instructions}\n\n${message.content}` : message.content;
                continue;
            }
        }

        if (message.role !== 'system') {
            sawNonSystemInput = true;
        }

        if (Array.isArray(message.tool_calls)) {
            if (message.content) {
                input.push(removeUndefinedValues({
                    type: 'message',
                    role: message.role,
                    content: convertOpenAIResponsesContent(message.content),
                }));
            }

            for (const toolCall of message.tool_calls) {
                if (toolCall?.type !== 'function') {
                    continue;
                }

                input.push(removeUndefinedValues({
                    type: 'function_call',
                    call_id: toolCall.id,
                    name: toolCall.function?.name,
                    arguments: toolCall.function?.arguments || '',
                    status: 'completed',
                }));
            }
            continue;
        }

        if (message.role === 'tool') {
            input.push(removeUndefinedValues({
                type: 'function_call_output',
                call_id: message.tool_call_id,
                output: message.content ?? '',
            }));
            continue;
        }

        if (message.content === undefined || message.content === null) {
            continue;
        }

        input.push(removeUndefinedValues({
            type: 'message',
            role: message.role,
            content: convertOpenAIResponsesContent(message.content),
        }));
    }

    if (input.length === 0) {
        input.push({ type: 'message', role: 'user', content: 'Continue.' });
    }

    return { instructions, input };
}

/**
 * Converts xAI Chat Completions messages to native Responses input items.
 * Unlike OpenAI, xAI keeps system messages in the full input history.
 * @param {object[]} messages Chat Completions messages
 * @returns {object[]} Responses input
 */
function convertXAIResponsesInput(messages) {
    if (!Array.isArray(messages)) {
        return [{ role: 'user', content: String(messages ?? '') }];
    }

    const input = [];
    const toolCallIds = new Map();
    let toolCallCounter = 0;
    const getResponsesToolCallId = (id) => {
        if (!id || typeof id !== 'string' || id.startsWith('fc_')) {
            return id;
        }

        const mapped = toolCallIds.get(id);
        if (mapped) {
            return mapped;
        }

        const responsesId = `fc_mapped_${++toolCallCounter}`;
        toolCallIds.set(id, responsesId);
        return responsesId;
    };

    for (const message of messages) {
        if (!message || typeof message !== 'object') {
            continue;
        }

        if (Array.isArray(message.tool_calls)) {
            if (message.content) {
                input.push(removeUndefinedValues({
                    role: message.role,
                    content: convertOpenAIResponsesContent(message.content),
                }));
            }

            for (const toolCall of message.tool_calls) {
                if (toolCall?.type !== 'function') {
                    continue;
                }

                const callId = getResponsesToolCallId(toolCall.id);
                input.push(removeUndefinedValues({
                    type: 'function_call',
                    id: callId,
                    call_id: callId,
                    name: toolCall.function?.name,
                    arguments: toolCall.function?.arguments || '',
                    status: 'completed',
                }));
            }
            continue;
        }

        if (message.role === 'tool') {
            input.push(removeUndefinedValues({
                type: 'function_call_output',
                call_id: getResponsesToolCallId(message.tool_call_id),
                output: message.content ?? '',
            }));
            continue;
        }

        if (message.content === undefined || message.content === null) {
            continue;
        }

        input.push(removeUndefinedValues({
            role: message.role,
            content: convertOpenAIResponsesContent(message.content),
        }));
    }

    if (input.length === 0) {
        input.push({ role: 'user', content: 'Continue.' });
    }

    return input;
}

/**
 * Converts Chat Completions tool definitions to Responses tool definitions.
 * @param {object[]} tools Chat Completions tool definitions
 * @returns {object[]|undefined} Responses tool definitions
 */
function convertOpenAIResponsesTools(tools) {
    if (!Array.isArray(tools) || tools.length === 0) {
        return undefined;
    }

    return tools.map(tool => {
        if (tool?.type === 'function' && tool.function) {
            return removeUndefinedValues({
                type: 'function',
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
                strict: tool.function.strict,
            });
        }

        return tool;
    });
}

/**
 * Converts Chat Completions tool_choice to Responses tool_choice.
 * @param {string|object} toolChoice Chat Completions tool choice
 * @returns {string|object|undefined} Responses-compatible tool choice
 */
function convertOpenAIResponsesToolChoice(toolChoice) {
    if (!toolChoice || typeof toolChoice === 'string') {
        return toolChoice;
    }

    if (toolChoice.type === 'function' && toolChoice.function?.name) {
        return { type: 'function', name: toolChoice.function.name };
    }

    return toolChoice;
}

/**
 * Reads the Meta Model API key from saved secrets, falling back to the official MODEL_API_KEY env var.
 * @param {import('express').Request} request Express request
 * @returns {string|undefined} API key
 */
function getMetaApiKey(request) {
    return readSecret(request.user.directories, SECRET_KEYS.META, request.body.secret_id) || process.env.MODEL_API_KEY;
}

/**
 * Builds a native Meta Responses request body.
 * @param {import('express').Request} request Express request
 * @param {object} bodyParams Provider-specific body params
 * @param {{instructions: string|undefined, input: object[]}} responsesInput Responses input
 * @returns {object} Responses request body
 */
function buildMetaResponsesRequestBody(request, bodyParams, responsesInput) {
    const text = {};
    if (request.body.json_schema) {
        text.format = removeUndefinedValues({
            type: 'json_schema',
            name: request.body.json_schema.name,
            description: request.body.json_schema.description,
            strict: request.body.json_schema.strict ?? true,
            schema: request.body.json_schema.value,
        });
    }

    const reasoning = {};
    if (request.body.reasoning_effort && request.body.reasoning_effort !== 'none') {
        reasoning.effort = request.body.reasoning_effort;
    }
    if (request.body.include_reasoning) {
        const summaryProfile = ['auto', 'concise', 'detailed'].includes(request.body.reasoning_summary)
            ? request.body.reasoning_summary
            : 'auto';
        reasoning.summary = summaryProfile;
    }

    const tools = convertOpenAIResponsesTools(bodyParams.tools) || [];
    if (request.body.enable_web_search) {
        tools.push({ type: 'web_search' });
    }

    return removeUndefinedValues({
        model: request.body.model || 'muse-spark-1.1',
        instructions: responsesInput.instructions,
        input: responsesInput.input,
        store: false,
        stream: request.body.stream,
        temperature: request.body.temperature,
        top_p: request.body.top_p,
        frequency_penalty: request.body.frequency_penalty,
        presence_penalty: request.body.presence_penalty,
        max_output_tokens: request.body.max_completion_tokens ?? request.body.max_tokens,
        tools: tools.length ? tools : undefined,
        tool_choice: tools.length ? 'auto' : undefined,
        text: Object.keys(text).length ? text : undefined,
        reasoning: Object.keys(reasoning).length ? reasoning : undefined,
    });
}

/**
 * Removes opaque encrypted reasoning content from Meta Responses objects.
 * @param {any} value Response object or nested value
 * @returns {any} Sanitized value
 */
function stripMetaEncryptedReasoning(value) {
    if (Array.isArray(value)) {
        for (const item of value) {
            stripMetaEncryptedReasoning(item);
        }
        return value;
    }

    if (!value || typeof value !== 'object') {
        return value;
    }

    if (value.type === 'reasoning') {
        delete value.encrypted_content;
    }

    for (const item of Object.values(value)) {
        stripMetaEncryptedReasoning(item);
    }

    return value;
}

/**
 * Strips encrypted reasoning content from a single SSE event chunk.
 * @param {string} event Server-sent event chunk
 * @returns {string} Sanitized event chunk
 */
function sanitizeMetaResponsesSseEvent(event) {
    return event.split(/\r?\n/).map(line => {
        const match = line.match(/^(data:\s?)(.*)$/);
        if (!match || match[2] === '[DONE]') {
            return line;
        }

        const data = tryParse(match[2]);
        if (!data) {
            return line;
        }

        stripMetaEncryptedReasoning(data);
        return `${match[1]}${JSON.stringify(data)}`;
    }).join('\n');
}

/**
 * Pipes a Meta Responses stream while removing encrypted reasoning content.
 * @param {import('node-fetch').Response} from The Fetch API response to pipe from
 * @param {import('express').Response} to The Express response to pipe to
 * @returns {Promise<void>}
 */
async function forwardSanitizedMetaResponsesStream(from, to) {
    if (!from.ok || !from.body || !to.socket) {
        return forwardFetchResponse(from, to);
    }

    to.statusCode = from.status === 401 ? 400 : from.status;
    to.statusMessage = from.statusText;

    let pending = '';
    const sanitizer = new Transform({
        transform(chunk, _encoding, callback) {
            pending += chunk.toString('utf8');
            const events = pending.split(/\r?\n\r?\n/);
            pending = events.pop() || '';
            for (const event of events) {
                this.push(`${sanitizeMetaResponsesSseEvent(event)}\n\n`);
            }
            callback();
        },
        flush(callback) {
            if (pending) {
                this.push(sanitizeMetaResponsesSseEvent(pending));
            }
            callback();
        },
    });

    from.body.pipe(sanitizer).pipe(to);

    to.socket.on('close', function () {
        if (from.body) {
            from.body.destroy();
        }
        to.end();
    });

    from.body.on('end', function () {
        console.info('Streaming request finished');
        to.end();
    });
}

/**
 * Builds a native OpenAI Responses request body.
 * @param {import('express').Request} request Express request
 * @param {object} bodyParams Provider-specific body params
 * @param {{instructions: string|undefined, input: object[]}} responsesInput Responses input
 * @returns {object} Responses request body
 */
function buildOpenAIResponsesRequestBody(request, bodyParams, responsesInput) {
    const text = {};
    if (request.body.json_schema) {
        text.format = removeUndefinedValues({
            type: 'json_schema',
            name: request.body.json_schema.name,
            description: request.body.json_schema.description,
            strict: request.body.json_schema.strict ?? true,
            schema: request.body.json_schema.value,
        });
    }
    if (request.body.verbosity && OPENAI_VERBOSITY_MODELS.test(request.body.model)) {
        text.verbosity = request.body.verbosity;
    }

    const effort = getOpenAIReasoningEffort(request.body.model, request.body.reasoning_effort);
    const mode = getOpenAIReasoningMode(request.body.model, request.body.reasoning_mode);
    const reasoning = {};
    if (effort) {
        reasoning.effort = effort;
        reasoning.context = 'current_turn';
    }
    if (mode) {
        reasoning.mode = mode;
    }
    if (request.body.include_reasoning && OPENAI_REASONING_EFFORT_MODELS.includes(request.body.model)) {
        reasoning.summary = 'auto';
        reasoning.context = 'current_turn';
    }

    const include = [];
    if (bodyParams.top_logprobs) {
        include.push('message.output_text.logprobs');
    }

    const omitSampling = isOpenAIResponsesNoSamplingModel(request.body.model, effort);

    return removeUndefinedValues({
        model: request.body.model,
        instructions: responsesInput.instructions,
        input: responsesInput.input,
        store: false,
        stream: request.body.stream,
        temperature: omitSampling ? undefined : request.body.temperature,
        top_p: omitSampling ? undefined : request.body.top_p,
        max_output_tokens: request.body.max_completion_tokens ?? request.body.max_tokens,
        tools: convertOpenAIResponsesTools(bodyParams.tools),
        tool_choice: convertOpenAIResponsesToolChoice(bodyParams.tool_choice),
        text: Object.keys(text).length ? text : undefined,
        reasoning: Object.keys(reasoning).length ? reasoning : undefined,
        top_logprobs: bodyParams.top_logprobs,
        include: include.length ? include : undefined,
        prompt_cache_key: bodyParams.prompt_cache_key,
        user: bodyParams.user,
    });
}

/**
 * Builds a native xAI Responses request body.
 * @param {import('express').Request} request Express request
 * @param {object} bodyParams Provider-specific body params
 * @param {object[]} input Responses input
 * @returns {object} Responses request body
 */
function buildXAIResponsesRequestBody(request, bodyParams, input) {
    const text = {};
    if (request.body.json_schema) {
        text.format = removeUndefinedValues({
            type: 'json_schema',
            name: request.body.json_schema.name,
            description: request.body.json_schema.description,
            strict: request.body.json_schema.strict ?? true,
            schema: request.body.json_schema.value,
        });
    }

    const reasoning = {};
    if (XAI_REASONING_EFFORTS.has(request.body.reasoning_effort)) {
        reasoning.effort = request.body.reasoning_effort;
    }

    return removeUndefinedValues({
        model: request.body.model,
        input: input,
        store: false,
        stream: request.body.stream,
        temperature: request.body.temperature,
        top_p: request.body.top_p,
        max_output_tokens: request.body.max_completion_tokens ?? request.body.max_tokens,
        tools: convertOpenAIResponsesTools(bodyParams.tools),
        tool_choice: convertOpenAIResponsesToolChoice(bodyParams.tool_choice),
        text: Object.keys(text).length ? text : undefined,
        reasoning: Object.keys(reasoning).length ? reasoning : undefined,
        prompt_cache_key: bodyParams.prompt_cache_key,
        logprobs: bodyParams.logprobs,
        top_logprobs: bodyParams.top_logprobs,
    });
}

/**
 * Moves SillyTavern's internal reasoning field to Moonshot's reasoning_content field.
 * @param {object[]} messages Prompt messages
 * @param {boolean} thinkingEnabled Whether the request uses Moonshot thinking mode
 * @param {boolean} preservedThinking Whether assistant reasoning history should be retained
 * @returns {void}
 */
function normalizeMoonshotReasoningContent(messages, thinkingEnabled, preservedThinking) {
    if (!Array.isArray(messages)) {
        return;
    }

    for (const message of messages) {
        const hasToolCalls = Array.isArray(message.tool_calls);
        const hasReasoning = typeof message.reasoning === 'string' && message.reasoning.length > 0;
        const shouldIncludeReasoning = thinkingEnabled && (hasToolCalls || (preservedThinking && message.role === 'assistant' && hasReasoning));
        if (shouldIncludeReasoning && hasReasoning) {
            message.reasoning_content = message.reasoning;
        }
        delete message.reasoning;

        if (!shouldIncludeReasoning) {
            delete message.reasoning_content;
            continue;
        }

        if (thinkingEnabled && !('reasoning_content' in message)) {
            message.reasoning_content = '';
        }
    }
}

/**
 * Module-scoped Claude caching configuration values.
 */
const cacheTTL = getConfigValue('claude.extendedTTL', false, 'boolean') ? '1h' : '5m';
const enableAutomaticPromptCache = getConfigValue('claude.enableAutomaticPromptCache', false, 'boolean');
const enableSystemPromptCache = getConfigValue('claude.enableSystemPromptCache', false, 'boolean');
const cachingAtDepth = (() => {
    const value = getConfigValue('claude.cachingAtDepth', -1, 'number');
    return Number.isInteger(value) && value >= 0 ? value : -1;
})();
const enableAdaptiveThinking = getConfigValue('claude.enableAdaptiveThinking', true, 'boolean');
let warnedAutomaticCacheBreakpointLimit = false;

/**
 * Adds top-level automatic caching when an explicit breakpoint slot remains.
 * @param {object} requestBody Anthropic-compatible request body
 */
function addAutomaticClaudeCacheControl(requestBody) {
    if (!enableAutomaticPromptCache) {
        return;
    }

    let explicitBreakpoints = 0;
    for (const section of [requestBody.system, requestBody.tools]) {
        if (Array.isArray(section)) {
            explicitBreakpoints += section.filter(block => block?.cache_control).length;
        }
    }
    for (const message of requestBody.messages || []) {
        explicitBreakpoints += message?.cache_control ? 1 : 0;
        if (Array.isArray(message?.content)) {
            explicitBreakpoints += message.content.filter(block => block?.cache_control).length;
        }
    }

    if (explicitBreakpoints >= 4) {
        if (!warnedAutomaticCacheBreakpointLimit) {
            console.warn('Claude automatic prompt caching disabled for requests already using four explicit cache breakpoints.');
            warnedAutomaticCacheBreakpointLimit = true;
        }
        return;
    }

    requestBody.cache_control = cacheTTL === '1h'
        ? { type: 'ephemeral', ttl: '1h' }
        : { type: 'ephemeral' };
}

/**
 * Cache for cacheable (writing) OpenRouter model IDs.
 * @type {string[]}
 */
const openRouterCacheableModels = [];

/**
 * Checks if an OpenRouter model supports prompt cache writing.
 * Uses a cache to avoid repeated API calls.
 * @param {string} modelId - The OpenRouter model ID
 * @returns {Promise<boolean>} `true` if the model supports writing cache
 */
async function isOpenRouterModelCacheable(modelId) {
    if (openRouterCacheableModels.includes(modelId)) {
        return true;
    }

    try {
        const response = await fetch(`${API_OPENROUTER}/models`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            signal: AbortSignal.timeout(5000),
        });

        if (!response.ok) {
            console.warn(`OpenRouter models API returned ${response.status}: ${response.statusText}`);
            return false;
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            console.warn('OpenRouter API response format unexpected');
            return false;
        }

        const model = data.data.find(m => m.id === modelId);
        const supportsCache = model?.pricing?.input_cache_write != null;

        if (supportsCache) {
            openRouterCacheableModels.push(modelId);
        }

        return supportsCache;
    } catch (error) {
        console.warn(`Failed to check OpenRouter cache support for ${modelId}:`, error.message);
        return false;
    }
}

/**
 * Gets OpenRouter transforms based on the request.
 * @param {import('express').Request} request Express request
 * @returns {string[] | undefined} OpenRouter transforms
 */
function getOpenRouterTransforms(request) {
    switch (request.body.middleout) {
        case 'on':
            return ['middle-out'];
        case 'off':
            return [];
        case 'auto':
            return undefined;
    }
}

/**
 * Gets OpenRouter server tools based on the request.
 * @param {import('express').Request} request
 * @returns {any[]} OpenRouter server tools
 */
function getOpenRouterServerTools(request) {
    const tools = [];

    if (request.body.enable_web_search) {
        tools.push({ 'type': 'openrouter:web_search' });
    }

    return tools;
}

/**
 * Gets Requesty server tools based on the request.
 * @param {import('express').Request} request
 * @returns {any[]} Requesty server tools
 */
function getRequestyServerTools(request) {
    const tools = [];

    if (request.body.enable_web_search) {
        tools.push({ type: 'web_search' });
    }

    return tools;
}

/**
 * Hacky way to use JSON schema only if json_object format is supported.
 * @param {object} bodyParams Additional body parameters
 * @param {object[]} messages Array of messages
 * @param {object} jsonSchema JSON schema object
 */
function setJsonObjectFormat(bodyParams, messages, jsonSchema) {
    bodyParams['response_format'] = {
        type: 'json_object',
    };
    const message = {
        role: 'user',
        content: `JSON schema for the response:\n${JSON.stringify(jsonSchema.value, null, 4)}`,
    };
    messages.push(message);
}

/**
 * Sends a request to Claude API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendClaudeRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_CLAUDE).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.CLAUDE, request.body.secret_id);
    const divider = '-'.repeat(process.stdout.columns);

    if (!apiKey) {
        console.warn(color.red(`Claude API key is missing.\n${divider}`));
        return response.status(400).send({ error: true });
    }

    try {
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });
        const additionalHeaders = {};
        const betaHeaders = [];
        const useTools = Array.isArray(request.body.tools) && request.body.tools.length > 0;
        const useSystemPrompt = Boolean(request.body.use_sysprompt);
        const useMidConversationSystemMessages = supportsClaudeMidConversationSystemMessages(request.body.model);
        const convertedPrompt = convertClaudeMessages(request.body.messages, request.body.assistant_prefill, useSystemPrompt, useTools, getPromptNames(request), useMidConversationSystemMessages);
        const useThinking = isClaudeThinkingModel(request.body.model);
        const useWebSearch = /^claude-(3-5|3-7|opus-4|opus-5|sonnet-4|sonnet-5|haiku-4-5|opus-4-5|opus-4-6|sonnet-4-6|opus-4-7|opus-4-8|fable-5|mythos-5)/.test(request.body.model) && Boolean(request.body.enable_web_search);
        const isLimitedSampling = isClaudeLimitedSamplingModel(request.body.model);
        const useVerbosity = isClaudeVerbosityModel(request.body.model);
        const noPrefillModel = isClaudeNoPrefillModel(request.body.model);
        const isAdaptiveModel = isClaudeAdaptiveThinkingModel(request.body.model, enableAdaptiveThinking);
        const noSamplingModel = isClaudeNoSamplingModel(request.body.model);
        const forcedAdaptiveModel = isClaudeForcedAdaptiveThinkingModel(request.body.model);
        const omittedThinkingDisplayModel = noSamplingModel;
        let fixThinkingPrefill = false;
        // Add custom stop sequences
        const stopSequences = [];
        if (Array.isArray(request.body.stop)) {
            stopSequences.push(...request.body.stop);
        }

        const requestBody = {
            /** @type {any} */ system: [],
            messages: convertedPrompt.messages,
            model: request.body.model,
            max_tokens: request.body.max_tokens,
            stop_sequences: stopSequences,
            temperature: request.body.temperature,
            top_p: request.body.top_p,
            top_k: request.body.top_k,
            stream: request.body.stream,
        };
        if (useSystemPrompt) {
            if (enableSystemPromptCache && Array.isArray(convertedPrompt.systemPrompt) && convertedPrompt.systemPrompt.length) {
                convertedPrompt.systemPrompt[convertedPrompt.systemPrompt.length - 1].cache_control = { type: 'ephemeral', ttl: cacheTTL };
            }

            requestBody.system = convertedPrompt.systemPrompt;
        } else {
            delete requestBody.system;
        }
        if (useTools) {
            betaHeaders.push('tools-2024-05-16');
            requestBody.tool_choice = { type: request.body.tool_choice };
            requestBody.tools = request.body.tools
                .filter(tool => tool.type === 'function')
                .map(tool => tool.function)
                .map(fn => ({ name: fn.name, description: fn.description, input_schema: flattenSchema(fn.parameters, request.body.chat_completion_source) }));

            if (enableSystemPromptCache && requestBody.tools.length) {
                requestBody.tools[requestBody.tools.length - 1].cache_control = { type: 'ephemeral', ttl: cacheTTL };
            }
        }

        // Structured output is a forced tool
        if (request.body.json_schema) {
            const jsonTool = {
                name: request.body.json_schema.name,
                description: request.body.json_schema.description || 'Well-formed JSON object',
                input_schema: request.body.json_schema.value,
            };
            requestBody.tools = [...(requestBody.tools || []), jsonTool];
            requestBody.tool_choice = { type: 'tool', name: request.body.json_schema.name };
        }

        if (useWebSearch) {
            const webSearchTool = [{
                'type': 'web_search_20250305',
                'name': 'web_search',
            }];
            requestBody.tools = [...webSearchTool, ...(requestBody.tools || [])];
        }

        if (cachingAtDepth !== -1) {
            cachingAtDepthForClaude(convertedPrompt.messages, cachingAtDepth, cacheTTL);
        }

        if (enableSystemPromptCache || cachingAtDepth !== -1) {
            betaHeaders.push('prompt-caching-2024-07-31');
            betaHeaders.push('extended-cache-ttl-2025-04-11');
        }

        if (isLimitedSampling) {
            if (requestBody.top_p < 1) {
                delete requestBody.temperature;
            } else {
                delete requestBody.top_p;
            }
        }

        if (noSamplingModel) {
            delete requestBody.temperature;
            delete requestBody.top_p;
            delete requestBody.top_k;
        }

        const reasoningEffort = request.body.reasoning_effort;
        const includeReasoning = Boolean(request.body.include_reasoning);
        const disableThinking = shouldDisableClaudeThinking(request.body.model, request.body.claude_disable_thinking);
        const budgetTokens = calculateClaudeBudgetTokens(requestBody.max_tokens, reasoningEffort, requestBody.stream, isAdaptiveModel);

        // Adaptive thinking: returns a string effort level (like Gemini 3)
        if (disableThinking) {
            requestBody.thinking = { type: 'disabled' };
        } else if (useThinking && typeof budgetTokens === 'string') {
            fixThinkingPrefill = true;
            requestBody.thinking = { type: 'adaptive' };
            if (omittedThinkingDisplayModel && includeReasoning) {
                requestBody.thinking.display = 'summarized';
            }
            requestBody.output_config ??= {};
            requestBody.output_config.effort = budgetTokens;
            // top_k is not allowed in adaptive mode
            delete requestBody.top_k;
        } else if (useThinking && budgetTokens === null && (forcedAdaptiveModel || (noSamplingModel && includeReasoning))) {
            // Adaptive-only models use adaptive thinking without explicit effort at auto.
            fixThinkingPrefill = true;
            requestBody.thinking = { type: 'adaptive' };
            if (includeReasoning) {
                requestBody.thinking.display = 'summarized';
            }
            // top_k is not allowed in adaptive mode
            delete requestBody.top_k;
        } else if (useThinking && Number.isInteger(budgetTokens)) {
            // Traditional thinking: returns a numeric budget
            fixThinkingPrefill = true;
            const minThinkTokens = 1024;
            if (requestBody.max_tokens <= minThinkTokens) {
                const newValue = requestBody.max_tokens + minThinkTokens;
                console.warn(color.yellow(`Claude thinking requires a minimum of ${minThinkTokens} response tokens.`));
                console.info(color.blue(`Increasing response length to ${newValue}.`));
                requestBody.max_tokens = newValue;
            }
            requestBody.thinking = {
                type: 'enabled',
                budget_tokens: budgetTokens,
            };

            // NO I CAN'T SILENTLY IGNORE THE TEMPERATURE.
            delete requestBody.temperature;
            delete requestBody.top_p;
            delete requestBody.top_k;
        }

        if ((fixThinkingPrefill || noPrefillModel) && convertedPrompt.messages.length && convertedPrompt.messages[convertedPrompt.messages.length - 1].role === 'assistant') {
            convertedPrompt.messages[convertedPrompt.messages.length - 1].role = 'user';
        }

        // Verbosity = 'effort' (same values as OpenAI) - only if not already set by adaptive thinking
        if (useVerbosity && request.body.verbosity && !requestBody.output_config?.effort) {
            betaHeaders.push('effort-2025-11-24');
            requestBody.output_config ??= {};
            requestBody.output_config.effort = request.body.verbosity;
        }

        if (betaHeaders.length) {
            additionalHeaders['anthropic-beta'] = betaHeaders.join(',');
        }

        addAutomaticClaudeCacheControl(requestBody);
        console.debug('Claude request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/messages', {
            method: 'POST',
            signal: controller.signal,
            body: JSON.stringify(requestBody),
            headers: {
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'x-api-key': apiKey,
                ...additionalHeaders,
            },
        });

        if (request.body.stream) {
            // Pipe remote SSE stream to Express response
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const generateResponseText = await generateResponse.text();
                console.warn(color.red(`Claude API returned error: ${generateResponse.status} ${generateResponse.statusText}\n${generateResponseText}\n${divider}`));
                return response.status(500).send({ error: true });
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();
            const responseText = Array.isArray(generateResponseJson?.content)
                ? generateResponseJson.content.filter(block => block?.type === 'text').map(block => block.text).join('')
                : '';
            console.debug('Claude response:', generateResponseJson);

            // Wrap it back to OAI format + save the original content
            const reply = { choices: [{ 'message': { 'content': responseText } }], content: generateResponseJson.content };
            return response.send(reply);
        }
    } catch (error) {
        console.error(color.red(`Error communicating with Claude: ${error}\n${divider}`));
        if (!response.headersSent) {
            return response.status(500).send({ error: true });
        }
    }
}

/**
 * Sends a request to Google AI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMakerSuiteRequest(request, response) {
    const useVertexAi = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.VERTEXAI;
    const apiName = useVertexAi ? 'Google Vertex AI' : 'Google AI Studio';
    let apiUrl;
    let apiKey;

    let authHeader;
    let authType;

    if (useVertexAi) {
        apiUrl = new URL(request.body.reverse_proxy || API_VERTEX_AI);

        try {
            const auth = await getVertexAIAuth(request);
            authHeader = auth.authHeader;
            authType = auth.authType;
            console.debug(`Using Vertex AI authentication type: ${authType}`);
        } catch (error) {
            console.warn(`${apiName} authentication failed: ${error.message}`);
            return response.status(400).send({ error: true, message: error.message });
        }
    } else {
        apiUrl = new URL(request.body.reverse_proxy || API_MAKERSUITE);
        apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE, request.body.secret_id);

        if (!request.body.reverse_proxy && !apiKey) {
            console.warn(`${apiName} API key is missing.`);
            return response.status(400).send({ error: true });
        }

        authHeader = `Bearer ${apiKey}`;
        authType = 'api_key';
    }

    const model = String(request.body.model);
    const stream = Boolean(request.body.stream);
    const enableWebSearch = Boolean(request.body.enable_web_search);
    const requestImages = Boolean(request.body.request_images);
    const reasoningEffort = String(request.body.reasoning_effort);
    const includeReasoning = Boolean(request.body.include_reasoning);
    const aspectRatio = String(request.body.request_image_aspect_ratio);
    const imageSize = String(request.body.request_image_resolution);
    const isGemma3 = /gemma-3/.test(model);
    const isLearnLM = model.includes('learnlm');

    const responseMimeType = request.body.responseMimeType ?? (request.body.json_schema ? 'application/json' : undefined);
    const responseSchema = request.body.responseSchema ?? (request.body.json_schema ? request.body.json_schema.value : undefined);
    const omitSampling = isGeminiNoSamplingModel(model);

    const generationConfig = {
        stopSequences: request.body.stop,
        candidateCount: omitSampling ? undefined : 1,
        maxOutputTokens: request.body.max_tokens,
        temperature: omitSampling ? undefined : request.body.temperature,
        topP: omitSampling ? undefined : request.body.top_p,
        topK: omitSampling ? undefined : request.body.top_k || undefined,
        responseMimeType: responseMimeType,
        responseSchema: responseSchema,
        seed: request.body.seed,
    };

    function getGeminiBody() {
        // #region UGLY MODEL LISTS AREA
        const imageGenerationModels = [
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-exp-image-generation',
            'gemini-2.0-flash-preview-image-generation',
            'gemini-2.5-flash-image-preview',
            'gemini-2.5-flash-image',
            'gemini-3-pro-image-preview',
            'gemini-3.1-flash-image-preview',
        ];

        const isThinkingConfigModel = m => (/^gemini-2.5-(flash|pro)/.test(m) && !/-image(-preview)?$/.test(m)) || (/^gemini-3[.\d]*-(flash|pro)/.test(m));
        const isImageSizeModel = m => /^gemini-3/.test(m);

        const noSearchModels = [
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash-lite-001',
            'gemini-2.0-flash-lite-preview-02-05',
            'gemini-robotics-er-1.5-preview',
        ];
        // #endregion

        if (!Array.isArray(generationConfig.stopSequences) || !generationConfig.stopSequences.length) {
            delete generationConfig.stopSequences;
        }

        const enableImageModality = requestImages && imageGenerationModels.includes(model);
        const enableImageConfig = enableImageModality && (aspectRatio || imageSize);
        if (enableImageModality) {
            generationConfig.responseModalities = ['text', 'image'];
            if (enableImageConfig) {
                generationConfig.imageConfig = {};
                if (imageSize && isImageSizeModel(model)) {
                    generationConfig.imageConfig.imageSize = imageSize;
                }
                if (aspectRatio) {
                    generationConfig.imageConfig.aspectRatio = aspectRatio;
                }
            }
        }

        const useSystemPrompt = !enableImageModality && !isGemma3 && request.body.use_sysprompt;

        const tools = [];
        const prompt = convertGooglePrompt(request.body.messages, model, useSystemPrompt, getPromptNames(request));
        const safetySettings = [...GEMINI_SAFETY, ...(useVertexAi ? VERTEX_SAFETY : [])];

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0 && !enableImageModality && !isGemma3) {
            const functionDeclarations = [];
            const customTools = [];
            for (const tool of request.body.tools) {
                if (tool.type === 'function') {
                    if (tool.function.parameters?.$schema) {
                        delete tool.function.parameters.$schema;
                    }
                    if (tool.function.parameters?.properties && Object.keys(tool.function.parameters.properties).length === 0) {
                        delete tool.function.parameters;
                    }
                    functionDeclarations.push(tool.function);
                } else if (tool[tool.type]) {
                    customTools.push({ [tool.type]: tool[tool.type] });
                }
            }
            if (functionDeclarations.length > 0) {
                tools.push({ function_declarations: functionDeclarations });
            }
            // Custom tools are only supported when no function calling is present
            if (functionDeclarations.length === 0 && customTools.length > 0) {
                tools.push(...customTools);
            }
        }

        if (enableWebSearch && !enableImageModality && !isGemma3 && !isLearnLM && !noSearchModels.includes(model)) {
            // Tool use with function calling is unsupported
            if (!tools.some(t => t.function_declarations)) {
                tools.push({ google_search: {} });
            }
        }

        if (isThinkingConfigModel(model)) {
            const thinkingConfig = { includeThoughts: includeReasoning };

            const thinkingBudget = calculateGoogleBudgetTokens(generationConfig.maxOutputTokens, reasoningEffort, model);
            if (typeof thinkingBudget === 'number' && Number.isInteger(thinkingBudget)) {
                thinkingConfig.thinkingBudget = thinkingBudget;
            }

            if (typeof thinkingBudget === 'string' && thinkingBudget.length > 0) {
                thinkingConfig.thinkingLevel = thinkingBudget;
            }

            // Vertex doesn't allow mixing disabled thinking with includeThoughts
            if (useVertexAi && thinkingBudget === 0 && thinkingConfig.includeThoughts) {
                console.info('Thinking budget is 0, but includeThoughts is true. Thoughts will not be included in the response.');
                thinkingConfig.includeThoughts = false;
            }

            generationConfig.thinkingConfig = thinkingConfig;
        }

        let body = {
            contents: prompt.contents,
            safetySettings: safetySettings,
            generationConfig: generationConfig,
        };

        if (useSystemPrompt && Array.isArray(prompt.system_instruction.parts) && prompt.system_instruction.parts.length) {
            body.systemInstruction = prompt.system_instruction;
        }

        if (tools.length) {
            body.tools = tools;

            const toolChoice = request.body.tool_choice;
            let functionCallingConfig;

            // Translate OpenAI's `tool_choice` to Gemini's `functionCallingConfig`
            if (typeof toolChoice === 'string') {
                switch (toolChoice) {
                    case 'none':
                        functionCallingConfig = { mode: 'NONE' };
                        break;
                    case 'required':
                        functionCallingConfig = { mode: 'ANY' };
                        break;
                    case 'auto':
                        functionCallingConfig = { mode: 'AUTO' };
                        break;
                }
            } else if (typeof toolChoice === 'object' && toolChoice?.function?.name) {
                // Force a specific function call
                functionCallingConfig = {
                    mode: 'ANY',
                    allowedFunctionNames: [toolChoice.function.name],
                };
            }

            if (functionCallingConfig) {
                body.toolConfig = { functionCallingConfig };
            }
        }

        return body;
    }

    const body = getGeminiBody();
    console.debug(`${apiName} request:`, body);

    try {
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        const apiVersion = getConfigValue('gemini.apiVersion', 'v1beta');
        const responseType = (stream ? 'streamGenerateContent' : 'generateContent');

        let url;
        let headers = {
            'Content-Type': 'application/json',
        };

        if (useVertexAi) {
            if (authType === 'express') {
                // For Express mode (API key authentication), use the key parameter
                const keyParam = authHeader.replace('Bearer ', '');
                const region = request.body.vertexai_region || 'us-central1';
                const projectId = request.body.vertexai_express_project_id;
                const baseUrl = region === 'global'
                    ? 'https://aiplatform.googleapis.com'
                    : `https://${region}-aiplatform.googleapis.com`;
                url = projectId
                    ? `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}?key=${keyParam}${stream ? '&alt=sse' : ''}`
                    : `${baseUrl}/v1/publishers/google/models/${model}:${responseType}?key=${keyParam}${stream ? '&alt=sse' : ''}`;
            } else if (authType === 'full') {
                // For Full mode (service account authentication), use project-specific URL
                // Get project ID from Service Account JSON
                const serviceAccountJson = readSecret(request.user.directories, SECRET_KEYS.VERTEXAI_SERVICE_ACCOUNT, request.body.secret_id);
                if (!serviceAccountJson) {
                    console.warn('Vertex AI Service Account JSON is missing.');
                    return response.status(400).send({ error: true });
                }

                let projectId;
                try {
                    const serviceAccount = JSON.parse(serviceAccountJson);
                    projectId = getProjectIdFromServiceAccount(serviceAccount);
                } catch (error) {
                    console.error('Failed to extract project ID from Service Account JSON:', error);
                    return response.status(400).send({ error: true });
                }
                const region = request.body.vertexai_region || 'us-central1';
                // Handle global region differently - no region prefix in hostname
                if (region === 'global') {
                    url = `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                } else {
                    url = `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                }
                headers['Authorization'] = authHeader;
            } else {
                // For proxy mode, use the original URL with Authorization header
                url = `${apiUrl.toString().replace(/\/$/, '')}/v1/publishers/google/models/${model}:${responseType}${stream ? '?alt=sse' : ''}`;
                headers['Authorization'] = authHeader;
            }
        } else {
            url = `${apiUrl.toString().replace(/\/$/, '')}/${apiVersion}/models/${model}:${responseType}?key=${apiKey}${stream ? '&alt=sse' : ''}`;
        }

        const generateResponse = await fetch(url, {
            body: JSON.stringify(body),
            method: 'POST',
            headers: headers,
            signal: controller.signal,
        });

        if (stream) {
            try {
                // Pipe remote SSE stream to Express response
                await forwardFetchResponse(generateResponse, response);
            } catch (error) {
                console.error('Error forwarding streaming response:', error);
                if (!response.headersSent) {
                    return response.status(500).send({ error: true });
                }
            }
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`${apiName} API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }

            /** @type {any} */
            const generateResponseJson = await generateResponse.json();

            const candidates = generateResponseJson?.candidates;
            if (!candidates || candidates.length === 0) {
                let message = `${apiName} API returned no candidate`;
                console.warn(message, generateResponseJson);
                if (generateResponseJson?.promptFeedback?.blockReason) {
                    message += `\nPrompt was blocked due to : ${generateResponseJson.promptFeedback.blockReason}`;
                }
                return response.send({ error: { message } });
            }

            const responseContent = candidates[0].content ?? candidates[0].output;
            const functionCall = (candidates?.[0]?.content?.parts ?? []).some(part => part.functionCall);
            const inlineData = (candidates?.[0]?.content?.parts ?? []).some(part => part.inlineData);
            console.debug(`${apiName} response:`, util.inspect(generateResponseJson, { depth: 5, colors: true }));

            const responseText = typeof responseContent === 'string' ? responseContent : responseContent?.parts?.filter(part => !part.thought)?.map(part => part.text)?.join('\n\n');
            if (!responseText && !functionCall && !inlineData) {
                let message = `${apiName} Candidate text empty`;
                console.warn(message, generateResponseJson);
                return response.send({ error: { message } });
            }

            // Wrap it back to OAI format (responseContent includes thought signatures in parts array)
            const reply = { choices: [{ 'message': { 'content': responseText } }], responseContent };
            return response.send(reply);
        }
    } catch (error) {
        console.error(`Error communicating with ${apiName} API:`, error);
        if (!response.headersSent) {
            return response.status(500).send({ error: true });
        }
    }
}

/**
 * Sends a request to AI21 API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendAI21Request(request, response) {
    if (!request.body) return response.sendStatus(400);

    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AI21, request.body.secret_id);
    if (!apiKey) {
        console.warn('AI21 API key is missing.');
        return response.status(400).send({ error: true });
    }

    const bodyParams = {};
    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });
    // Hack to support JSON schema
    if (request.body.json_schema) {
        bodyParams.response_format = {
            type: 'json_object',
        };
        const message = {
            role: 'user',
            content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
        };
        request.body.messages.push(message);
    }
    const convertedPrompt = convertAI21Messages(request.body.messages, getPromptNames(request));
    const body = {
        messages: convertedPrompt,
        model: request.body.model,
        max_tokens: request.body.max_tokens,
        temperature: request.body.temperature,
        top_p: request.body.top_p,
        stop: request.body.stop,
        stream: request.body.stream,
        tools: request.body.tools,
        ...bodyParams,
    };
    const options = {
        method: 'POST',
        headers: {
            accept: 'application/json',
            'content-type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
    };

    console.debug('AI21 request:', body);

    try {
        const generateResponse = await fetch(API_AI21 + '/chat/completions', options);
        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`AI21 API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('AI21 response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with AI21 API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to MistralAI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMistralAIRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI, request.body.secret_id);

    if (!apiKey) {
        console.warn('MistralAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    try {
        const messages = convertMistralMessages(request.body.messages, getPromptNames(request));
        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        const requestBody = {
            'model': request.body.model,
            'messages': messages,
            'temperature': request.body.temperature,
            'top_p': request.body.top_p,
            'frequency_penalty': request.body.frequency_penalty,
            'presence_penalty': request.body.presence_penalty,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'safe_prompt': request.body.safe_prompt,
            'random_seed': request.body.seed === -1 ? undefined : request.body.seed,
            'reasoning_effort': ['high', 'none'].includes(request.body.reasoning_effort) ? request.body.reasoning_effort : undefined,
            'stop': Array.isArray(request.body.stop) && request.body.stop.length > 0 ? request.body.stop : undefined,
        };

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            requestBody['tools'] = request.body.tools;
            requestBody['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.json_schema) {
            requestBody['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
            timeout: 0,
        };

        console.debug('MisralAI request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);
        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`MistralAI API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('MistralAI response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with MistralAI API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Cohere API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendCohereRequest(request, response) {
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE, request.body.secret_id);
    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    if (!apiKey) {
        console.warn('Cohere API key is missing.');
        return response.status(400).send({ error: true });
    }

    try {
        const convertedHistory = convertCohereMessages(request.body.messages, getPromptNames(request));
        const tools = [];

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            tools.push(...request.body.tools);
            tools.forEach(tool => {
                if (tool?.function?.parameters?.$schema) {
                    delete tool.function.parameters.$schema;
                }
            });
        }

        // https://docs.cohere.com/reference/chat
        const requestBody = {
            stream: Boolean(request.body.stream),
            model: request.body.model,
            messages: convertedHistory.chatHistory,
            temperature: request.body.temperature,
            max_tokens: request.body.max_tokens,
            k: request.body.top_k,
            p: request.body.top_p,
            seed: request.body.seed,
            stop_sequences: request.body.stop,
            frequency_penalty: request.body.frequency_penalty,
            presence_penalty: request.body.presence_penalty,
            documents: [],
            tools: tools,
        };

        const canDoSafetyMode = String(request.body.model).endsWith('08-2024');
        if (canDoSafetyMode) {
            requestBody.safety_mode = 'OFF';
        }

        if (request.body.json_schema) {
            requestBody.response_format = {
                type: 'json_schema',
                schema: request.body.json_schema.value,
            };
        }

        console.debug('Cohere request:', requestBody);

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
            timeout: 0,
        };

        const apiUrl = API_COHERE_V2 + '/chat';

        if (request.body.stream) {
            const stream = await fetch(apiUrl, config);
            await forwardFetchResponse(stream, response);
        } else {
            const generateResponse = await fetch(apiUrl, config);
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`Cohere API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Cohere response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with Cohere API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to DeepSeek API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendDeepSeekRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK, request.body.secret_id);

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('DeepSeek API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;

            // DeepSeek doesn't permit empty required arrays
            bodyParams.tools.forEach(tool => {
                const required = tool?.function?.parameters?.required;
                if (Array.isArray(required) && required.length === 0) {
                    delete tool.function.parameters.required;
                }
            });
        }

        // Hack to support JSON schema
        if (request.body.json_schema) {
            bodyParams.response_format = {
                type: 'json_object',
            };
            const message = {
                role: 'user',
                content: `JSON schema for the response:\n${JSON.stringify(request.body.json_schema.value, null, 4)}`,
            };
            request.body.messages.push(message);
        }

        const processedMessages = addAssistantPrefix(postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.SEMI_TOOLS, getPromptNames(request)), bodyParams.tools, 'prefix');
        addReasoningContentToToolCalls(processedMessages);

        if (request.body.include_reasoning && request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        const requestBody = {
            'messages': processedMessages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'stop': request.body.stop,
            'seed': request.body.seed,
            'thinking': { type: request.body.include_reasoning ? 'enabled' : 'disabled' },
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('DeepSeek request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`DeepSeek API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('DeepSeek response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with DeepSeek API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to XAI API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendXaiRequest(request, response) {
    const apiUrl = new URL(request.body.reverse_proxy || API_XAI).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI, request.body.secret_id);
    const useXAIResponsesApi = request.body.xai_api_type === 'responses';

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('xAI API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        if (XAI_REASONING_EFFORTS.has(request.body.reasoning_effort)) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        if (request.body.enable_web_search && useXAIResponsesApi) {
            bodyParams['tools'] = [{ type: 'web_search' }, ...(bodyParams['tools'] || [])];
        } else if (request.body.enable_web_search) {
            bodyParams['search_parameters'] = { mode: 'on' };
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    strict: request.body.json_schema.strict ?? true,
                    schema: request.body.json_schema.value,
                },
            };
        }

        const promptCacheKey = getXaiPromptCacheKey(request);
        if (promptCacheKey && useXAIResponsesApi) {
            bodyParams['prompt_cache_key'] = promptCacheKey;
        }

        const processedMessages = request.body.messages = convertXAIMessages(request.body.messages, getPromptNames(request));
        const requestBody = useXAIResponsesApi
            ? buildXAIResponsesRequestBody(request, bodyParams, convertXAIResponsesInput(processedMessages))
            : {
                'messages': processedMessages,
                'model': request.body.model,
                'temperature': request.body.temperature,
                'max_tokens': request.body.max_tokens,
                'max_completion_tokens': request.body.max_completion_tokens,
                'stream': request.body.stream,
                'presence_penalty': request.body.presence_penalty,
                'frequency_penalty': request.body.frequency_penalty,
                'top_p': request.body.top_p,
                'seed': request.body.seed,
                'n': request.body.n,
                ...bodyParams,
            };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        if (promptCacheKey && !useXAIResponsesApi) {
            config.headers['x-grok-conv-id'] = promptCacheKey;
        }

        console.debug('xAI request:', requestBody);

        const generateResponse = await fetch(apiUrl + (useXAIResponsesApi ? '/responses' : '/chat/completions'), config);

        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`xAI API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('xAI response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with xAI API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to AI/ML API.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendAimlapiRequest(request, response) {
    const apiUrl = API_AIMLAPI;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI, request.body.secret_id);

    if (!apiKey) {
        console.warn('AI/ML API key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'seed': request.body.seed,
            'n': request.body.n,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
                ...AIMLAPI_HEADERS,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('AI/ML API request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn(`AI/ML API returned error: ${generateResponse.status} ${generateResponse.statusText} ${errorText}`);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('AI/ML API response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with AI/ML API: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Electron Hub.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendElectronHubRequest(request, response) {
    const apiUrl = API_ELECTRONHUB;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.ELECTRONHUB, request.body.secret_id);

    if (!apiKey) {
        console.warn('Electron Hub key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (request.body.enable_web_search) {
            bodyParams['web_search'] = true;
        }

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.reasoning_effort) {
            bodyParams['reasoning_effort'] = request.body.reasoning_effort;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const isClaude = /^claude-/.test(request.body.model);

        if (Array.isArray(request.body.messages) && isClaude) {
            if (enableSystemPromptCache) {
                cachingSystemPromptForOpenRouter(request.body.messages, cacheTTL);
            }

            if (cachingAtDepth !== -1) {
                cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, cacheTTL);
            }
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'top_p': request.body.top_p,
            'top_k': request.body.top_k,
            'logit_bias': request.body.logit_bias,
            'seed': request.body.seed,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('Electron Hub request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn('Electron Hub returned error: ', errorText);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Electron Hub response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with Electron Hub: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to Chutes.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendChutesRequest(request, response) {
    const apiUrl = API_CHUTES;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.CHUTES, request.body.secret_id);

    if (!apiKey) {
        console.warn('Chutes key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        let bodyParams = {};

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        if (request.body.logprobs > 0) {
            bodyParams['top_logprobs'] = request.body.logprobs;
            bodyParams['logprobs'] = true;
        }

        if (request.body.json_schema) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    description: request.body.json_schema.description,
                    schema: request.body.json_schema.value,
                    strict: request.body.json_schema.strict ?? true,
                },
            };
        }

        const requestBody = {
            'messages': request.body.messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.max_tokens,
            'stream': request.body.stream,
            'presence_penalty': request.body.presence_penalty,
            'frequency_penalty': request.body.frequency_penalty,
            'repetition_penalty': request.body.repetition_penalty,
            'min_p': request.body.min_p,
            'top_p': request.body.top_p,
            'top_k': request.body.top_k,
            'seed': request.body.seed,
            'stop': request.body.stop,
            'reasoning_effort': request.body.reasoning_effort,
            'logit_bias': request.body.logit_bias,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('Chutes request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn('Chutes returned error: ', errorText);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('Chutes response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with Chutes: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * Sends a request to MiniMax.
 * @param {express.Request} request Express request
 * @param {express.Response} response Express response
 */
async function sendMinimaxRequest(request, response) {
    const defaultApiUrl = request.body.minimax_endpoint === MINIMAX_ENDPOINT.CN
        ? API_MINIMAX_CN : API_MINIMAX;
    const apiUrl = new URL(request.body.reverse_proxy || defaultApiUrl).toString();
    const apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MINIMAX, request.body.secret_id);

    if (!apiKey && !request.body.reverse_proxy) {
        console.warn('MiniMax key is missing.');
        return response.status(400).send({ error: true });
    }

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', function () {
        controller.abort();
    });

    try {
        // MiniMax does not allow consecutive messages with the same role.
        // Merge them into a single message to avoid "invalid chat setting (2013)".
        const messages = postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.MERGE_TOOLS, getPromptNames(request));

        let bodyParams = {};

        if (Array.isArray(request.body.tools) && request.body.tools.length > 0) {
            bodyParams['tools'] = request.body.tools;
            bodyParams['tool_choice'] = request.body.tool_choice;
        }

        const requestBody = {
            'messages': messages,
            'model': request.body.model,
            'temperature': request.body.temperature,
            'max_tokens': request.body.model === 'M2-her' ? Math.min(request.body.max_tokens, 2048) : request.body.max_tokens,
            'stream': request.body.stream,
            'top_p': request.body.top_p,
            'stop': request.body.stop,
            ...bodyParams,
        };

        const config = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('MiniMax request:', requestBody);

        const generateResponse = await fetch(apiUrl + '/chat/completions', config);

        if (request.body.stream) {
            await forwardFetchResponse(generateResponse, response);
        } else {
            if (!generateResponse.ok) {
                const errorText = await generateResponse.text();
                console.warn('MiniMax returned error: ', errorText);
                const errorJson = tryParse(errorText) ?? { error: true };
                return response.status(500).send(errorJson);
            }
            const generateResponseJson = await generateResponse.json();
            console.debug('MiniMax response:', generateResponseJson);
            return response.send(generateResponseJson);
        }
    } catch (error) {
        console.error('Error communicating with MiniMax: ', error);
        if (!response.headersSent) {
            response.send({ error: true });
        } else {
            response.end();
        }
    }
}

/**
 * @param {express.Request} request Express request object (contains request.body with all generate_data)
 * @param {express.Response} response Express response object
 */
async function sendAzureOpenAIRequest(request, response) {
    // 1. GATHER & VALIDATE SETTINGS
    const { azure_base_url, azure_deployment_name, azure_api_version } = request.body;
    const apiKey = readSecret(request.user.directories, SECRET_KEYS.AZURE_OPENAI, request.body.secret_id);
    if (!azure_base_url || !azure_deployment_name || !azure_api_version || !apiKey) {
        return response.status(400).send({
            error: {
                message: 'Azure OpenAI configuration is incomplete. Please provide Base URL, Deployment Name, API Version, and API Key in the connection settings.',
            },
        });
    }

    // 2. PREPARE THE REQUEST
    const url = new URL(`/openai/deployments/${azure_deployment_name}/chat/completions`, azure_base_url);
    url.searchParams.set('api-version', azure_api_version);
    const endpointUrl = url.toString();

    // Create the base payload with all standard parameters
    const apiRequestBody = /** @type {any} */ ({});
    for (const key of AZURE_OPENAI_KEYS) {
        if (Object.hasOwn(request.body, key)) {
            apiRequestBody[key] = request.body[key];
        }
    }

    // Handle Structured Output (JSON Mode) by translating the custom `json_schema` object.
    if (request.body.json_schema) {
        apiRequestBody['response_format'] = {
            type: 'json_schema',
            json_schema: {
                name: request.body.json_schema.name,
                strict: request.body.json_schema.strict ?? true,
                schema: request.body.json_schema.value,
            },
        };
    }

    // Adjust logprobs for Azure OpenAI, which follows the OpenAI Chat Completions API spec.
    if (typeof apiRequestBody.logprobs === 'number' && apiRequestBody.logprobs > 0) {
        apiRequestBody.top_logprobs = apiRequestBody.logprobs;
        apiRequestBody.logprobs = true;
    }

    // Do not send reasoning effort to models which do not support it
    apiRequestBody['reasoning_effort'] = getOpenAIReasoningEffort(request.body.model, request.body.reasoning_effort);

    const controller = new AbortController();
    request.socket.removeAllListeners('close');
    request.socket.on('close', () => controller.abort());

    const config = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'api-key': apiKey,
        },
        body: JSON.stringify(apiRequestBody),
        signal: controller.signal,
    };

    console.info(`Sending request to Azure OpenAI: ${endpointUrl}`);
    console.debug('Azure OpenAI Request Body:', apiRequestBody);
    try {
        const fetchResponse = await fetch(endpointUrl, config);

        if (request.body.stream) {
            return await forwardFetchResponse(fetchResponse, response);
        }

        if (fetchResponse.ok) {
            /** @type {any} */
            const json = await fetchResponse.json();
            console.debug('Azure OpenAI response:', json);
            return response.send(json);
        }

        const text = await fetchResponse.text();
        const data = tryParse(text) || { error: { message: fetchResponse.statusText || 'Unknown error occurred' } };
        return response.status(500).send(data);
    } catch (error) {
        const message = error.name === 'AbortError'
            ? 'Request was aborted by the client.'
            : (error.message || 'An unknown network error occurred.');
        return response.status(500).send({ error: { message, ...error } });
    }
}

export const router = express.Router();

router.post('/status', async function (request, statusResponse) {
    try {
        if (!request.body) return statusResponse.sendStatus(400);

        let apiUrl = '';
        let apiKey = '';
        let headers = {};
        let queryParams = {};

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_OPENAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AGENTROUTER) {
            apiUrl = API_AGENTROUTER;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.AGENTROUTER, request.body.secret_id);
            headers = { ...AGENTROUTER_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
            apiUrl = 'https://openrouter.ai/api/v1';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER, request.body.secret_id);
            // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
            headers = getOpenRouterHeaders(request.body);
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.REQUESTY) {
            apiUrl = API_REQUESTY;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.REQUESTY, request.body.secret_id);
            headers = { ...REQUESTY_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_MISTRAL).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MISTRALAI, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            apiUrl = request.body.custom_url;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM, request.body.secret_id);
            headers = {};
            mergeObjectWithYaml(headers, request.body.custom_include_headers);
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE) {
            apiUrl = API_COHERE_V1;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COHERE, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CHUTES) {
            apiUrl = API_CHUTES;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CHUTES, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ELECTRONHUB) {
            apiUrl = API_ELECTRONHUB;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.ELECTRONHUB, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
            apiUrl = API_NANOGPT;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT, request.body.secret_id);
            headers = {};
            queryParams = { detailed: true };
            if (['favorites', 'mostused'].includes(String(request.body.sort_models))) {
                queryParams.sort = request.body.sort_models;
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.DEEPSEEK) {
            apiUrl = new URL(request.body.reverse_proxy || API_DEEPSEEK.replace('/beta', '')).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.DEEPSEEK, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.XAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_XAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.XAI, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.META) {
            apiUrl = API_META;
            apiKey = getMetaApiKey(request);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AIMLAPI) {
            apiUrl = API_AIMLAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.AIMLAPI, request.body.secret_id);
            headers = { ...AIMLAPI_HEADERS };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
            const isAnonymous = request.body.pollinations_endpoint === POLLINATIONS_ENDPOINT.ANONYMOUS;
            apiUrl = 'https://gen.pollinations.ai/text';
            apiKey = isAnonymous ? 'anonymous' : readSecret(request.user.directories, SECRET_KEYS.POLLINATIONS, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
            apiUrl = API_GROQ;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NVIDIA) {
            apiUrl = API_NVIDIA;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NVIDIA, request.body.secret_id);
            headers = { 'Accept': 'application/json' };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
            apiUrl = API_COMETAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI, request.body.secret_id);
            headers = {};
            throw new Error('This provider is temporarily disabled.');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
            apiUrl = new URL(request.body.reverse_proxy || API_MOONSHOT).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MOONSHOT, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
            apiUrl = API_FIREWORKS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS, request.body.secret_id);
            headers = {};
            if (!apiKey) {
                console.warn('Fireworks AI API key is missing.');
                return statusResponse.status(400).send({ error: true });
            }
            try {
                const { models, fallback } = await fetchFireworksServerlessModels(apiKey);
                console.debug('Available Fireworks AI models:', models.map(m => m.id));
                return statusResponse.send({ data: models, fallback });
            } catch (error) {
                console.warn('Fireworks AI models endpoint failed:', error.message || error);
                const status = error['status'] || 500;
                return statusResponse.status(status).send({ error: true });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MAKERSUITE) {
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MAKERSUITE, request.body.secret_id);
            apiUrl = trimTrailingSlash(request.body.reverse_proxy || API_MAKERSUITE);
            const apiVersion = getConfigValue('gemini.apiVersion', 'v1beta');
            const modelsUrl = !apiKey && request.body.reverse_proxy
                ? `${apiUrl}/${apiVersion}/models`
                : `${apiUrl}/${apiVersion}/models?key=${apiKey}`;

            if (!apiKey && !request.body.reverse_proxy) {
                console.warn('Google AI Studio API key is missing.');
                return statusResponse.status(400).send({ error: true });
            }

            try {
                const response = await fetch(modelsUrl);

                if (response.ok) {
                    /** @type {any} */
                    const data = await response.json();
                    // Transform Google AI Studio models to OpenAI format
                    const models = data.models
                        ?.filter(model => model.supportedGenerationMethods?.includes('generateContent'))
                        ?.map(model => ({
                            ...model,
                            id: model.name.replace('models/', ''),
                        })) || [];

                    console.info('Available Google AI Studio models:', models.map(m => m.id));
                    return statusResponse.send({ data: models });
                } else {
                    console.warn('Google AI Studio models endpoint failed:', response.status, response.statusText);
                    return statusResponse.send({ error: true, bypass: true, data: { data: [] } });
                }
            } catch (error) {
                console.error('Error fetching Google AI Studio models:', error);
                return statusResponse.send({ error: true, bypass: true, data: { data: [] } });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AZURE_OPENAI) {
            const { azure_base_url, azure_deployment_name, azure_api_version } = request.body;
            const apiKey = readSecret(request.user.directories, SECRET_KEYS.AZURE_OPENAI, request.body.secret_id);

            // 1) Validate configuration from the frontend
            if (!apiKey || !azure_base_url || !azure_deployment_name || !azure_api_version) {
                console.warn('Azure OpenAI status check failed: missing config from frontend.');
                return statusResponse.status(400).send({ error: true, message: 'Azure configuration is incomplete.' });
            }
            // 2) Build URLs using the URL API for consistency and robustness.
            const modelsUrl = new URL('/openai/models', azure_base_url);
            modelsUrl.searchParams.set('api-version', azure_api_version);

            const chatUrl = new URL(`/openai/deployments/${azure_deployment_name}/chat/completions`, azure_base_url);
            chatUrl.searchParams.set('api-version', azure_api_version);

            // Map common status codes to user-friendly error messages
            const azureStatusErrorMap = {
                400: 'API version may be invalid for this resource.',
                401: 'Invalid API key or insufficient permissions.',
                403: 'Invalid API key or insufficient permissions.',
                404: 'Endpoint URL appears incorrect (404).',
            };

            try {
                // ---- A) GET /models: fast sanity check for endpoint + api key + api version ----
                const apiConfigTest = await fetch(modelsUrl, {
                    method: 'GET',
                    headers: { 'api-key': apiKey, 'Accept': 'application/json' },
                });

                if (!apiConfigTest.ok) {
                    let errText = '';
                    try { errText = await apiConfigTest.text(); } catch { /* response body may be empty */ }

                    console.warn('Azure OpenAI GET /models failed:', apiConfigTest.status, apiConfigTest.statusText, errText || '');

                    const defaultMessage = `Azure Models endpoint error: ${apiConfigTest.statusText}`;
                    const message = azureStatusErrorMap[apiConfigTest.status] ?? defaultMessage;
                    return statusResponse.status(apiConfigTest.status).send({ error: true, message });
                }

                // ---- B) POST /chat/completions: verify deployment + read underlying model ID ----
                // Small, deterministic probe to minimize cost/latency
                const modelPayload = {
                    messages: [{ role: 'user', content: 'Say word Hi' }],
                    stream: false,
                    max_completion_tokens: 5,
                };

                const modelRequest = await fetch(chatUrl, {
                    method: 'POST',
                    headers: { 'api-key': apiKey, 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify(modelPayload),
                });

                let modelResponse;
                try {
                    modelResponse = await modelRequest.json();
                } catch {
                    modelResponse = { raw: 'Failed to parse JSON response from chat completions probe.' };
                }

                const modelId = /** @type {any} */ (modelResponse)?.model;
                if (!modelId) {
                    console.warn('Azure status check succeeded but could not find a model ID in the response.');
                    console.debug('Azure Response Body:', modelResponse);
                    // Keep a benign success to avoid UX disruption in the UI
                    return statusResponse.send({ data: [] });
                }

                console.info(color.green('Azure OpenAI connection successful. Detected model:'), modelId);
                // Consistent response format: always an array of { id }
                return statusResponse.send({ data: [{ id: modelId }] });
            } catch (error) {
                console.error('Azure OpenAI status check connection error:', error);
                return statusResponse.status(500).send({ error: true, message: 'Failed to connect to the Azure endpoint.' });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.SILICONFLOW) {
            const defaultApiUrl = request.body.siliconflow_endpoint === SILICONFLOW_ENDPOINT.CN
                ? API_SILICONFLOW_CN : API_SILICONFLOW;
            apiUrl = defaultApiUrl;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.SILICONFLOW, request.body.secret_id);
            headers = {};
            queryParams = { type: 'text', sub_type: 'chat' };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ATLASCLOUD) {
            apiUrl = API_ATLASCLOUD;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.ATLASCLOUD, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MINIMAX) {
            const defaultApiUrl = request.body.minimax_endpoint === MINIMAX_ENDPOINT.CN
                ? API_MINIMAX_CN : API_MINIMAX;
            apiUrl = new URL(request.body.reverse_proxy || defaultApiUrl).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MINIMAX, request.body.secret_id);
            headers = {};
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.WORKERS_AI) {
            apiKey = readSecret(request.user.directories, SECRET_KEYS.WORKERS_AI, request.body.secret_id);

            if (!apiKey) {
                console.warn('Cloudflare Workers AI API key is missing.');
                return statusResponse.status(400).send({ error: true });
            }

            try {
                const accountId = String(request.body.workers_ai_account_id || '').trim();
                if (!accountId) {
                    console.warn('Cloudflare Workers AI Account ID is missing.');
                    return statusResponse.status(400).send({ error: true });
                }

                const modelsUrl = new URL(`${API_WORKERS_AI}/${encodeURIComponent(accountId)}/ai/models/search`);
                modelsUrl.searchParams.set('task', 'Text Generation');
                modelsUrl.searchParams.set('per_page', '1000');

                const response = await fetch(modelsUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': 'Bearer ' + apiKey,
                    },
                });

                if (response.ok) {
                    /** @type {any} */
                    const data = await response.json();
                    const models = Array.isArray(data?.result)
                        ? data.result.map(model => ({ ...model, id: model.name }))
                        : [];

                    console.debug('Available Cloudflare Workers AI models:', models.map(m => m.id));
                    return statusResponse.send({ data: models });
                } else {
                    console.warn('Cloudflare Workers AI models endpoint failed:', response.status, response.statusText);
                    return statusResponse.status(response.status).send({ error: true });
                }
            } catch (error) {
                console.error('Error fetching Cloudflare Workers AI models:', error);
                return statusResponse.status(500).send({ error: true });
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FEATHERLESS) {
            apiUrl = API_FEATHERLESS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FEATHERLESS, request.body.secret_id);
            headers = { ...FEATHERLESS_HEADERS };
        } else {
            console.warn('This chat completion source is not supported yet.');
            return statusResponse.status(400).send({ error: true });
        }

        if (!apiKey && !request.body.reverse_proxy && ![CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.NVIDIA].includes(request.body.chat_completion_source)) {
            console.warn('Chat Completion API key is missing.');
            return statusResponse.status(400).send({ error: true });
        }

        const modelsUrl = new URL(urlJoin(apiUrl, '/models'));
        Object.keys(queryParams).forEach(key => {
            modelsUrl.searchParams.append(key, queryParams[key]);
        });
        const response = await fetch(modelsUrl, {
            method: 'GET',
            headers: {
                ...(apiKey ? { 'Authorization': 'Bearer ' + apiKey } : {}),
                ...headers,
            },
        });

        if (response.ok) {
            /** @type {any} */
            let data = await response.json();

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS && Array.isArray(data)) {
                data = { data: data.map(model => ({ id: model.name, ...model })) };
            }

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CHUTES && Array.isArray(data?.data)) {
                data.data = data.data
                    .filter(model => model?.id)
                    .map(model => {
                        if (model.pricing?.prompt !== undefined && model.pricing?.completion !== undefined) {
                            return {
                                ...model,
                                pricing: {
                                    ...model.pricing,
                                    input: model.pricing.prompt,
                                    output: model.pricing.completion,
                                },
                            };
                        }
                        return model;
                    });
            }

            statusResponse.send(data);

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COHERE && Array.isArray(data?.models)) {
                data.data = data.models.map(model => ({ id: model.name, ...model }));
            }

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER && Array.isArray(data?.data)) {
                let models = [];

                data.data.forEach(model => {
                    const context_length = model.context_length;
                    const tokens_dollar = Number(1 / (1000 * model.pricing?.prompt));
                    const tokens_rounded = (Math.round(tokens_dollar * 1000) / 1000).toFixed(0);
                    models[model.id] = {
                        tokens_per_dollar: tokens_rounded + 'k',
                        context_length: context_length,
                    };
                });

                console.info('Available OpenRouter models:', models);
            } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MISTRALAI) {
                const models = data?.data;
                console.info(models);
            } else {
                const models = data?.data;

                if (Array.isArray(models)) {
                    const modelIds = models.filter(x => x && typeof x === 'object').map(x => x.id).sort();
                    console.info('Available models:', modelIds);
                } else {
                    console.warn('Chat Completion endpoint did not return a list of models.');
                }
            }
        } else {
            console.error('Chat Completion status check failed. Either Access Token is incorrect or API endpoint is down.');
            statusResponse.send({ error: true, data: { data: [] } });
        }
    } catch (e) {
        console.error(e);

        if (!statusResponse.headersSent) {
            statusResponse.send({ error: true });
        } else {
            statusResponse.end();
        }
    }
});

router.post('/bias', async function (request, response) {
    if (!request.body || !Array.isArray(request.body))
        return response.sendStatus(400);

    try {
        const result = {};
        const model = getTokenizerModel(String(request.query.model || ''));

        // no bias for claude
        if (model == 'claude') {
            return response.send(result);
        }

        let encodeFunction;

        if (sentencepieceTokenizers.includes(model)) {
            const tokenizer = getSentencepiceTokenizer(model);
            const instance = await tokenizer?.get();
            if (!instance) {
                console.error('Tokenizer not initialized:', model);
                return response.send({});
            }
            encodeFunction = (text) => new Uint32Array(instance.encodeIds(text));
        } else if (webTokenizers.includes(model)) {
            const tokenizer = getWebTokenizer(model);
            const instance = await tokenizer?.get();
            if (!instance) {
                console.warn('Tokenizer not initialized:', model);
                return response.send({});
            }
            encodeFunction = (text) => new Uint32Array(instance.encode(text));
        } else {
            const tokenizer = getTiktokenTokenizer(model);
            encodeFunction = (tokenizer.encode.bind(tokenizer));
        }

        for (const entry of request.body) {
            if (!entry || !entry.text) {
                continue;
            }

            try {
                const tokens = getEntryTokens(entry.text, encodeFunction);

                for (const token of tokens) {
                    result[token] = entry.value;
                }
            } catch {
                console.warn('Tokenizer failed to encode:', entry.text);
            }
        }

        // not needed for cached tokenizers
        //tokenizer.free();
        return response.send(result);

        /**
         * Gets tokenids for a given entry
         * @param {string} text Entry text
         * @param {(string) => Uint32Array} encode Function to encode text to token ids
         * @returns {Uint32Array} Array of token ids
         */
        function getEntryTokens(text, encode) {
            // Get raw token ids from JSON array
            if (text.trim().startsWith('[') && text.trim().endsWith(']')) {
                try {
                    const json = JSON.parse(text);
                    if (Array.isArray(json) && json.every(x => typeof x === 'number')) {
                        return new Uint32Array(json);
                    }
                } catch {
                    // ignore
                }
            }

            // Otherwise, get token ids from tokenizer
            return encode(text);
        }
    } catch (error) {
        console.error(error);
        return response.send({});
    }
});

router.post('/generate', async function (request, response) {
    try {
        if (!request.body) return response.status(400).send({ error: true });

        const postProcessingType = request.body.custom_prompt_post_processing;
        if (Array.isArray(request.body.messages) && postProcessingType) {
            console.info('Applying custom prompt post-processing of type', postProcessingType);
            request.body.messages = postProcessPrompt(
                request.body.messages,
                postProcessingType,
                getPromptNames(request));
        }

        if (request.body.json_schema?.value) {
            request.body.json_schema.value = flattenSchema(request.body.json_schema.value, request.body.chat_completion_source);
        }

        switch (request.body.chat_completion_source) {
            case CHAT_COMPLETION_SOURCES.CLAUDE: return await sendClaudeRequest(request, response);
            case CHAT_COMPLETION_SOURCES.AI21: return await sendAI21Request(request, response);
            case CHAT_COMPLETION_SOURCES.MAKERSUITE: return await sendMakerSuiteRequest(request, response);
            case CHAT_COMPLETION_SOURCES.VERTEXAI: return await sendMakerSuiteRequest(request, response);
            case CHAT_COMPLETION_SOURCES.MISTRALAI: return await sendMistralAIRequest(request, response);
            case CHAT_COMPLETION_SOURCES.COHERE: return await sendCohereRequest(request, response);
            case CHAT_COMPLETION_SOURCES.DEEPSEEK: return await sendDeepSeekRequest(request, response);
            case CHAT_COMPLETION_SOURCES.AIMLAPI: return await sendAimlapiRequest(request, response);
            case CHAT_COMPLETION_SOURCES.XAI: return await sendXaiRequest(request, response);
            case CHAT_COMPLETION_SOURCES.CHUTES: return await sendChutesRequest(request, response);
            case CHAT_COMPLETION_SOURCES.MINIMAX: return await sendMinimaxRequest(request, response);
            case CHAT_COMPLETION_SOURCES.ELECTRONHUB: return await sendElectronHubRequest(request, response);
            case CHAT_COMPLETION_SOURCES.AZURE_OPENAI: return await sendAzureOpenAIRequest(request, response);
        }

        let apiUrl;
        let apiKey;
        let headers;
        let bodyParams;
        const isTextCompletion = Boolean(request.body.model && TEXT_COMPLETION_MODELS.includes(request.body.model)) || typeof request.body.messages === 'string';
        const useOpenAIResponsesApi = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI && request.body.openai_api_type === 'responses';
        const useMetaResponsesApi = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.META;

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENAI) {
            apiUrl = new URL(request.body.reverse_proxy || API_OPENAI).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.OPENAI, request.body.secret_id);
            headers = {};
            bodyParams = {
                logprobs: request.body.logprobs,
                top_logprobs: undefined,
            };

            // Adjust logprobs params for Chat Completions API, which expects { top_logprobs: number; logprobs: boolean; }
            if (!isTextCompletion && bodyParams.logprobs > 0) {
                bodyParams.top_logprobs = bodyParams.logprobs;
                bodyParams.logprobs = true;
            }

            if (getConfigValue('openai.randomizeUserId', false, 'boolean')) {
                bodyParams['user'] = uuidv4();
            }

            const promptCacheKey = getOpenAIPromptCacheKey(request);
            if (promptCacheKey && useOpenAIResponsesApi) {
                bodyParams['prompt_cache_key'] = promptCacheKey;
            }

            embedOpenRouterMedia(request.body.messages, { audio: true, video: false });
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.AGENTROUTER) {
            apiUrl = API_AGENTROUTER;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.AGENTROUTER, request.body.secret_id);
            headers = { ...AGENTROUTER_HEADERS };
            bodyParams = {};
            embedOpenRouterMedia(request.body.messages, { audio: true, video: false });
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER) {
            apiUrl = 'https://openrouter.ai/api/v1';
            apiKey = readSecret(request.user.directories, SECRET_KEYS.OPENROUTER, request.body.secret_id);
            // OpenRouter needs to pass the Referer and X-Title: https://openrouter.ai/docs#requests
            headers = getOpenRouterHeaders(request.body);
            const includeReasoning = Boolean(request.body.include_reasoning);
            const isMoonshot = isOpenRouterMoonshotModel(request.body.model);
            bodyParams = {
                transforms: getOpenRouterTransforms(request),
                reasoning: {
                    exclude: !includeReasoning,
                },
            };

            const sessionId = getOpenRouterSessionId(request);
            if (sessionId) {
                bodyParams['session_id'] = sessionId;
            }

            if (request.body.logprobs > 0) {
                bodyParams['top_logprobs'] = request.body.logprobs;
                bodyParams['logprobs'] = true;
            }

            if (request.body.min_p !== undefined) {
                bodyParams['min_p'] = request.body.min_p;
            }

            if (request.body.top_a !== undefined) {
                bodyParams['top_a'] = request.body.top_a;
            }

            if (request.body.repetition_penalty !== undefined) {
                bodyParams['repetition_penalty'] = request.body.repetition_penalty;
            }

            if (Array.isArray(request.body.provider) && request.body.provider.length > 0) {
                bodyParams['provider'] = {
                    allow_fallbacks: request.body.allow_fallbacks ?? true,
                    order: request.body.provider ?? [],
                };
            }

            if (Array.isArray(request.body.quantizations) && request.body.quantizations.length > 0) {
                bodyParams['provider'] ??= {};
                bodyParams['provider']['quantizations'] = request.body.quantizations;
            }

            if (request.body.use_fallback) {
                bodyParams['route'] = 'fallback';
            }

            if (request.body.reasoning_effort) {
                bodyParams['reasoning']['effort'] = request.body.reasoning_effort;
            }

            if (request.body.verbosity) {
                bodyParams['verbosity'] = request.body.verbosity;
            }

            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        strict: request.body.json_schema.strict ?? true,
                        schema: request.body.json_schema.value,
                    },
                };
            }

            const isClaude = /^anthropic\/claude/.test(request.body.model);
            const isGemini = /google\/gemini/.test(request.body.model);
            const isCacheableGemini = isGemini && await isOpenRouterModelCacheable(request.body.model);
            const enableGeminiSystemPromptCache = getConfigValue('gemini.enableSystemPromptCache', false, 'boolean');

            if (Array.isArray(request.body.messages)) {
                embedOpenRouterMedia(request.body.messages, { audio: true, video: true });
                addOpenRouterSignatures(request.body.messages, request.body.model);

                if (isMoonshot) {
                    normalizeMoonshotReasoningContent(request.body.messages, true, Boolean(request.body.moonshot_preserved_thinking));
                }

                if (isClaude) {
                    if (enableSystemPromptCache) {
                        cachingSystemPromptForOpenRouter(request.body.messages, cacheTTL);
                    }

                    if (cachingAtDepth !== -1) {
                        cachingAtDepthForOpenRouterClaude(request.body.messages, cachingAtDepth, cacheTTL);
                    }
                }

                if (isCacheableGemini && enableGeminiSystemPromptCache) {
                    cachingSystemPromptForOpenRouter(request.body.messages);
                }

                if (isMoonshot
                    && request.body.moonshot_thinking_prefill
                    && !request.body.json_schema) {
                    extractMoonshotThinkingPrefill(request.body.messages);
                    addAssistantPrefix(request.body.messages, [], 'partial', true);
                }
            }

            if (isGemini) {
                bodyParams['safety_settings'] = GEMINI_SAFETY;
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.REQUESTY) {
            apiUrl = API_REQUESTY;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.REQUESTY, request.body.secret_id);
            headers = { ...REQUESTY_HEADERS };
            bodyParams = {};
            embedOpenRouterMedia(request.body.messages, { audio: true, video: false });
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            apiUrl = request.body.custom_url;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.CUSTOM, request.body.secret_id);
            headers = {};
            bodyParams = {
                logprobs: request.body.logprobs,
                top_logprobs: undefined,
            };

            // Adjust logprobs params for Chat Completions API, which expects { top_logprobs: number; logprobs: boolean; }
            if (!isTextCompletion && bodyParams.logprobs > 0) {
                bodyParams.top_logprobs = bodyParams.logprobs;
                bodyParams.logprobs = true;
            }

            mergeObjectWithYaml(bodyParams, request.body.custom_include_body);
            mergeObjectWithYaml(headers, request.body.custom_include_headers);
            embedOpenRouterMedia(request.body.messages, { audio: true, video: false });
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        strict: request.body.json_schema.strict ?? true,
                        schema: request.body.json_schema.value,
                    },
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.PERPLEXITY) {
            apiUrl = API_PERPLEXITY;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.PERPLEXITY, request.body.secret_id);
            headers = {};
            bodyParams = {
                reasoning_effort: request.body.reasoning_effort,
            };
            request.body.messages = postProcessPrompt(request.body.messages, PROMPT_PROCESSING_TYPE.STRICT, getPromptNames(request));
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        schema: request.body.json_schema.value,
                    },
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.GROQ) {
            apiUrl = API_GROQ;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.GROQ, request.body.secret_id);
            headers = {};
            bodyParams = {};
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        description: request.body.json_schema.description,
                        schema: request.body.json_schema.value,
                        strict: request.body.json_schema.strict ?? true,
                    },
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NVIDIA) {
            apiUrl = API_NVIDIA;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NVIDIA, request.body.secret_id);
            headers = { 'Accept': 'application/json' };
            const enabledParameters = new Set(Array.isArray(request.body.nvidia_enabled_parameters)
                ? request.body.nvidia_enabled_parameters
                : NVIDIA_DEFAULT_ENABLED_PARAMETERS);
            bodyParams = {};
            if (enabledParameters.has('thinking')) {
                bodyParams['thinking'] = {
                    type: request.body.include_reasoning ? 'enabled' : 'disabled',
                };
            }
            if (enabledParameters.has('reasoning_effort') && request.body.reasoning_effort) {
                bodyParams['reasoning_effort'] = request.body.reasoning_effort;
            }
            if (enabledParameters.has('min_p') && request.body.min_p !== undefined) {
                bodyParams['min_p'] = request.body.min_p;
            }
            if (enabledParameters.has('top_a') && request.body.top_a !== undefined) {
                bodyParams['top_a'] = request.body.top_a;
            }
            if (enabledParameters.has('repetition_penalty') && request.body.repetition_penalty !== undefined) {
                bodyParams['repetition_penalty'] = request.body.repetition_penalty;
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.META) {
            apiUrl = API_META;
            apiKey = getMetaApiKey(request);
            headers = {};
            bodyParams = {};
            if (!request.body.model) {
                request.body.model = 'muse-spark-1.1';
            }
            embedOpenRouterMedia(request.body.messages, { audio: false, video: true });
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
            apiUrl = API_FIREWORKS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FIREWORKS, request.body.secret_id);
            headers = {};
            bodyParams = {};
            if (request.body.model === FIREWORKS_LEGACY_DEFAULT_MODEL) {
                request.body.model = FIREWORKS_DEFAULT_MODEL;
            }
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: {
                        name: request.body.json_schema.name,
                        description: request.body.json_schema.description,
                        schema: request.body.json_schema.value,
                        strict: request.body.json_schema.strict ?? true,
                    },
                };
            }
            if (request.body.min_p !== undefined) {
                bodyParams['min_p'] = request.body.min_p;
            }
            if (request.body.typical_p !== undefined) {
                bodyParams['typical_p'] = request.body.typical_p;
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NANOGPT) {
            apiUrl = API_NANOGPT;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.NANOGPT, request.body.secret_id);
            headers = {};
            bodyParams = {};
            if (request.body.nanogpt_provider) {
                headers['X-Provider'] = request.body.nanogpt_provider;
            }
            if (request.body.nanogpt_payg_override) {
                headers['X-Billing-Mode'] = 'paygo';
                bodyParams['billing_mode'] = 'paygo';
            }
            if (request.body.nanogpt_service_tier) {
                const serviceTier = String(request.body.nanogpt_service_tier);
                if (['auto', 'default', 'flex', 'priority'].includes(serviceTier)) {
                    bodyParams['service_tier'] = serviceTier;
                }
            }
            if (request.body.enable_web_search && !/(?:^|:)online(?:[/:]|$)/.test(request.body.model)) {
                request.body.model = `${request.body.model}:online`;
            }
            if (request.body.min_p !== undefined) {
                bodyParams['min_p'] = request.body.min_p;
            }
            if (request.body.top_a !== undefined) {
                bodyParams['top_a'] = request.body.top_a;
            }
            if (request.body.repetition_penalty !== undefined) {
                bodyParams['repetition_penalty'] = request.body.repetition_penalty;
            }
            const reasoning = {};
            if (request.body.include_reasoning === false) {
                reasoning.exclude = true;
            }
            if (request.body.reasoning_effort && request.body.reasoning_effort !== 'auto') {
                const effort = NANOGPT_REASONING_EFFORT_MAP[request.body.reasoning_effort] ?? request.body.reasoning_effort;
                if (effort) {
                    reasoning.effort = effort;
                }
            }
            if (Object.keys(reasoning).length > 0) {
                bodyParams['reasoning'] = reasoning;
            }

            const isClaude = /(?:^|\/)claude[-_]/.test(request.body.model);
            if (enableSystemPromptCache && isClaude) {
                bodyParams['cache_control'] = {
                    'enabled': true,
                    'ttl': cacheTTL,
                };
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FEATHERLESS) {
            apiUrl = API_FEATHERLESS;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.FEATHERLESS, request.body.secret_id);
            headers = { ...FEATHERLESS_HEADERS };
            bodyParams = {
                min_p: request.body.min_p,
                repetition_penalty: request.body.repetition_penalty,
            };
            const isMoonshot = isFeatherlessMoonshotModel(request.body.model);
            const thinkingPrefill = isMoonshot && Boolean(request.body.moonshot_thinking_prefill) && !request.body.json_schema;
            const preservedThinking = isMoonshot && Boolean(request.body.moonshot_preserved_thinking);
            if (thinkingPrefill || preservedThinking) {
                normalizeMoonshotReasoningContent(request.body.messages, true, preservedThinking);
                if (thinkingPrefill) {
                    extractMoonshotThinkingPrefill(request.body.messages);
                    addAssistantPrefix(request.body.messages, [], 'partial', true);
                }
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.POLLINATIONS) {
            const isAnonymous = request.body.pollinations_endpoint === POLLINATIONS_ENDPOINT.ANONYMOUS;
            apiUrl = isAnonymous ? API_POLLINATIONS_ANON : API_POLLINATIONS;
            apiKey = isAnonymous ? 'anonymous' : readSecret(request.user.directories, SECRET_KEYS.POLLINATIONS, request.body.secret_id);
            headers = {};
            bodyParams = {
                seed: request.body.seed ?? Math.floor(Math.random() * 99999999),
            };
            if (!isAnonymous) {
                bodyParams['reasoning_effort'] = request.body.reasoning_effort;
                if (request.body.json_schema) {
                    bodyParams['response_format'] = {
                        type: 'json_schema',
                        json_schema: {
                            schema: request.body.json_schema.value,
                        },
                    };
                }
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT) {
            apiUrl = new URL(request.body.reverse_proxy || API_MOONSHOT).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.MOONSHOT, request.body.secret_id);
            headers = {};
            const isKimiK3 = isMoonshotKimiK3Model(request.body.model);
            const thinkingPrefill = Boolean(request.body.moonshot_thinking_prefill) && !request.body.json_schema;
            const preservedThinking = Boolean(request.body.moonshot_preserved_thinking);
            const thinkingEnabled = isMoonshotKimiAlwaysOnThinkingModel(request.body.model) || Boolean(request.body.include_reasoning) || thinkingPrefill || preservedThinking;
            bodyParams = isKimiK3 && thinkingEnabled
                ? {}
                : {
                    thinking: {
                        type: thinkingEnabled ? 'enabled' : 'disabled',
                    },
                };
            if (isKimiK3 && thinkingEnabled && request.body.reasoning_effort && request.body.reasoning_effort !== 'auto') {
                bodyParams.reasoning_effort = request.body.reasoning_effort;
            }
            normalizeMoonshotReasoningContent(request.body.messages, thinkingEnabled, preservedThinking);
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            } else {
                if (thinkingPrefill) {
                    extractMoonshotThinkingPrefill(request.body.messages);
                }
                addAssistantPrefix(request.body.messages, [], 'partial', true);
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.COMETAPI) {
            apiUrl = API_COMETAPI;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.COMETAPI, request.body.secret_id);
            headers = {};
            bodyParams = {
                reasoning_effort: request.body.reasoning_effort,
            };
            throw new Error('This provider is temporarily disabled.');
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ZAI) {
            const defaultApiUrl = request.body.zai_endpoint === ZAI_ENDPOINT.CODING ? API_ZAI_CODING : API_ZAI_COMMON;
            apiUrl = new URL(request.body.reverse_proxy || defaultApiUrl).toString();
            apiKey = request.body.reverse_proxy ? request.body.proxy_password : readSecret(request.user.directories, SECRET_KEYS.ZAI, request.body.secret_id);
            headers = {
                'Accept-Language': 'en-US,en',
            };
            bodyParams = {
                thinking: {
                    type: request.body.include_reasoning ? 'enabled' : 'disabled',
                },
            };
            if (request.body.include_reasoning && ['glm-5.2', 'glm-5.3'].includes(request.body.model) && request.body.reasoning_effort) {
                bodyParams['reasoning_effort'] = request.body.reasoning_effort;
            }
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.SILICONFLOW) {
            const defaultApiUrl = request.body.siliconflow_endpoint === SILICONFLOW_ENDPOINT.CN
                ? API_SILICONFLOW_CN : API_SILICONFLOW;
            apiUrl = defaultApiUrl;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.SILICONFLOW, request.body.secret_id);
            headers = {};
            bodyParams = {};
            if (request.body.json_schema) {
                setJsonObjectFormat(bodyParams, request.body.messages, request.body.json_schema);
            }
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ATLASCLOUD) {
            apiUrl = API_ATLASCLOUD;
            apiKey = readSecret(request.user.directories, SECRET_KEYS.ATLASCLOUD, request.body.secret_id);
            headers = {};
            bodyParams = {
                top_k: request.body.top_k,
                repetition_penalty: request.body.repetition_penalty,
            };
        } else if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.WORKERS_AI) {
            apiKey = readSecret(request.user.directories, SECRET_KEYS.WORKERS_AI, request.body.secret_id);
            const accountId = String(request.body.workers_ai_account_id || '').trim();
            if (!accountId) {
                console.warn('Cloudflare Workers AI Account ID is missing.');
                return response.status(400).send({ error: true });
            }
            apiUrl = `${API_WORKERS_AI}/${encodeURIComponent(accountId)}/ai/v1`;
            headers = {};
            bodyParams = {
                repetition_penalty: request.body.repetition_penalty,
            };
            if (request.body.json_schema) {
                bodyParams['response_format'] = {
                    type: 'json_schema',
                    json_schema: request.body.json_schema.value,
                };
            }
        } else {
            console.warn('This chat completion source is not supported yet.');
            return response.status(400).send({ error: true });
        }

        // Some OpenAI-compatible providers support reasoning effort.
        if (!useOpenAIResponsesApi && !useMetaResponsesApi && request.body.reasoning_effort && [CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI, CHAT_COMPLETION_SOURCES.AGENTROUTER, CHAT_COMPLETION_SOURCES.REQUESTY].includes(request.body.chat_completion_source)) {
            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.REQUESTY) {
                bodyParams['reasoning_effort'] = request.body.reasoning_effort;
            }
            if (OPENAI_REASONING_EFFORT_MODELS.includes(request.body.model)) {
                bodyParams['reasoning_effort'] = getOpenAIReasoningEffort(request.body.model, request.body.reasoning_effort);
            }
            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM && /^koboldcpp\/(.+)$/.test(request.body.model)) {
                bodyParams['reasoning_effort'] = request.body.reasoning_effort;
            }
        }

        if (!useOpenAIResponsesApi && !useMetaResponsesApi && request.body.verbosity && [CHAT_COMPLETION_SOURCES.CUSTOM, CHAT_COMPLETION_SOURCES.OPENAI].includes(request.body.chat_completion_source)) {
            if (OPENAI_VERBOSITY_MODELS.test(request.body.model)) {
                bodyParams['verbosity'] = request.body.verbosity;
            }
        }

        if (!apiKey && !request.body.reverse_proxy && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.CUSTOM) {
            console.warn('Chat Completion API key is missing.');
            return response.status(400).send({ error: true });
        }

        // Add custom stop sequences
        if (Array.isArray(request.body.stop) && request.body.stop.length > 0) {
            bodyParams['stop'] = request.body.stop;
        }

        const textPrompt = isTextCompletion ? convertTextCompletionPrompt(request.body.messages) : '';
        const endpointUrl = useOpenAIResponsesApi || useMetaResponsesApi ?
            `${apiUrl}/responses` :
            isTextCompletion && request.body.chat_completion_source !== CHAT_COMPLETION_SOURCES.OPENROUTER ?
            `${apiUrl}/completions` :
            `${apiUrl}/chat/completions`;

        const controller = new AbortController();
        request.socket.removeAllListeners('close');
        request.socket.on('close', function () {
            controller.abort();
        });

        if (!isTextCompletion) {
            const requestTools = Array.isArray(request.body.tools) ? request.body.tools : [];
            const openRouterTools = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER ? getOpenRouterServerTools(request) : [];
            const requestyTools = request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.REQUESTY ? getRequestyServerTools(request) : [];
            const tools = [...openRouterTools, ...requestyTools, ...requestTools];

            if (tools.length > 0) {
                bodyParams['tools'] = tools;
            }

            if (requestTools.length > 0) {
                bodyParams['tool_choice'] = request.body.tool_choice;
            }

            if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT
                && bodyParams.thinking?.type === 'enabled'
                && !['auto', 'none', undefined].includes(bodyParams['tool_choice'])) {
                bodyParams['tool_choice'] = 'auto';
            }
        }

        if (request.body.json_schema && !bodyParams['response_format']) {
            bodyParams['response_format'] = {
                type: 'json_schema',
                json_schema: {
                    name: request.body.json_schema.name,
                    strict: request.body.json_schema.strict ?? true,
                    schema: request.body.json_schema.value,
                },
            };
        }

        const responsesInput = useOpenAIResponsesApi || useMetaResponsesApi
            ? convertOpenAIResponsesInput(isTextCompletion ? textPrompt : request.body.messages)
            : undefined;
        let requestBody;
        if (useOpenAIResponsesApi) {
            requestBody = buildOpenAIResponsesRequestBody(request, bodyParams, responsesInput);
        } else if (useMetaResponsesApi) {
            requestBody = buildMetaResponsesRequestBody(request, bodyParams, responsesInput);
        } else {
            requestBody = {
                'messages': isTextCompletion === false ? request.body.messages : undefined,
                'prompt': isTextCompletion === true ? textPrompt : undefined,
                'model': request.body.model,
                'temperature': request.body.temperature,
                'max_tokens': request.body.max_tokens,
                'max_completion_tokens': request.body.max_completion_tokens,
                'stream': request.body.stream,
                'presence_penalty': request.body.presence_penalty,
                'frequency_penalty': request.body.frequency_penalty,
                'top_p': request.body.top_p,
                'top_k': request.body.top_k,
                'stop': isTextCompletion === false ? request.body.stop : undefined,
                'logit_bias': request.body.logit_bias,
                'seed': request.body.seed,
                'n': request.body.n,
                ...bodyParams,
            };
        }

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.OPENROUTER
            && /^anthropic\/claude/.test(request.body.model)) {
            addAutomaticClaudeCacheControl(requestBody);
        }

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.MOONSHOT && isMoonshotKimiFixedParameterModel(request.body.model)) {
            delete requestBody.temperature;
            delete requestBody.presence_penalty;
            delete requestBody.frequency_penalty;
            delete requestBody.top_p;
            delete requestBody.n;
        }

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.ATLASCLOUD) {
            sanitizeAtlascloudRequestBody(requestBody, request);
        }

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.FIREWORKS) {
            const sessionAffinity = getFireworksSessionAffinity(request);
            if (sessionAffinity) {
                headers['x-session-affinity'] = sessionAffinity;
            }
            sanitizeFireworksRequestBody(requestBody, request);
        }

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.NVIDIA) {
            const enabledParameters = new Set(Array.isArray(request.body.nvidia_enabled_parameters)
                ? request.body.nvidia_enabled_parameters
                : NVIDIA_DEFAULT_ENABLED_PARAMETERS);
            for (const parameter of ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'top_k', 'seed']) {
                if (!enabledParameters.has(parameter)) {
                    delete requestBody[parameter];
                }
            }
        }

        if (request.body.chat_completion_source === CHAT_COMPLETION_SOURCES.CUSTOM) {
            excludeKeysByYaml(requestBody, request.body.custom_exclude_body);
        }

        /** @type {import('node-fetch').RequestInit} */
        const config = {
            method: 'post',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + apiKey,
                ...headers,
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
        };

        console.debug('Chat Completion request:', requestBody);

        const fetchResponse = await fetch(endpointUrl, config);

        if (request.body.stream) {
            console.info('Streaming request in progress');
            if (useMetaResponsesApi) {
                return await forwardSanitizedMetaResponsesStream(fetchResponse, response);
            }
            return await forwardFetchResponse(fetchResponse, response);
        }

        if (fetchResponse.ok) {
            /** @type {any} */
            const json = await fetchResponse.json();
            if (useMetaResponsesApi) {
                stripMetaEncryptedReasoning(json);
            }
            console.debug('Chat Completion response:', json);
            return response.send(json);
        } else {
            const responseText = await fetchResponse.text();
            const errorData = tryParse(responseText);

            const message = fetchResponse.statusText || 'Unknown error occurred';
            const quota_error = fetchResponse.status === 429 && errorData?.error?.type === 'insufficient_quota';
            console.error('Chat completion request error: ', message, responseText);

            if (!response.headersSent) {
                response.send({ error: { message }, quota_error: quota_error });
            } else if (!response.writableEnded) {
                response.write(responseText);
            } else {
                response.end();
            }
        }
    } catch (error) {
        console.error('Generation failed', error);
        const message = error.code === 'ECONNREFUSED'
            ? `Connection refused: ${error.message}`
            : error.message || 'Unknown error occurred';

        if (!response.headersSent) {
            response.status(502).send({ error: { message, ...error } });
        } else {
            response.end();
        }
    }
});

const multimodalModels = express.Router();

multimodalModels.post('/pollinations', async (_req, res) => {
    try {
        const response = await fetch('https://gen.pollinations.ai/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data)) {
            return res.json([]);
        }

        const multimodalModels = data
            .filter(m => Array.isArray(m?.input_modalities))
            .filter(m => m.input_modalities.includes('image'))
            .map(m => m.name);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/aimlapi', async (_req, res) => {
    try {
        const response = await fetch('https://api.aimlapi.com/v1/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            return res.json([]);
        }

        const multimodalModels = data.data.filter(m => m?.features?.includes('openai/chat-completion.vision')).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/nanogpt', async (_req, res) => {
    try {
        const response = await fetch('https://nano-gpt.com/api/v1/models?detailed=true');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            return res.json([]);
        }

        const multimodalModels = data.data.filter(m => m?.capabilities?.vision).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/electronhub', async (_req, res) => {
    try {
        const response = await fetch('https://api.electronhub.ai/v1/models');

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const multimodalModels = data.data.filter(m => m.metadata?.vision).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/chutes', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.CHUTES);

        if (!key) {
            return res.json([]);
        }

        const response = await fetch('https://llm.chutes.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        const data = await response.json();

        const modelsData = /** @type {{object: string, data: Array<{id: string, input_modalities?: string[]}>}} */ (data);
        const multimodalModels = modelsData.data
            .filter(m => m.input_modalities?.includes('image'))
            .map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/mistral', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.MISTRALAI);

        if (!key) {
            return res.json([]);
        }

        const response = await fetch('https://api.mistral.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const multimodalModels = data.data.filter(m => m.capabilities?.vision).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/xai', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.XAI);

        if (!key) {
            return res.json([]);
        }

        // xAI's /models endpoint doesn't return modality info, so we must use /language-models instead
        const response = await fetch('https://api.x.ai/v1/language-models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const multimodalModels = data.models.filter(m => m.input_modalities?.includes('image')).map(m => m.id);
        if (!multimodalModels.includes('grok-4-0709')) {
            // The endpoint says it doesn't support images, but it does
            multimodalModels.push('grok-4-0709');
        }
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/moonshot', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.MOONSHOT);

        if (!key) {
            return res.json([]);
        }

        const response = await fetch('https://api.moonshot.ai/v1/models', {
            headers: {
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        const multimodalModels = data.data.filter(m => m.supports_image_in).map(m => m.id);
        return res.json(multimodalModels);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

multimodalModels.post('/workers_ai', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.WORKERS_AI);
        const accountId = String(req.body.workers_ai_account_id || '').trim();

        if (!key || !accountId) {
            return res.json([]);
        }

        const apiUrl = `https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(accountId)}/ai/models/search?task=Text+Generation&per_page=1000`;
        const response = await fetch(apiUrl, {
            method: 'GET',
            headers: { 'Authorization': 'Bearer ' + key },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const models = Array.isArray(data?.result)
            ? data.result
                .filter(m => Array.isArray(m.properties) && m.properties.some(p => p.property_id === 'vision' && p.value === 'true'))
                .map(m => m.name)
            : [];
        return res.json(models);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.use('/multimodal-models', multimodalModels);

router.post('/process', async function (request, response) {
    try {
        if (!Array.isArray(request.body.messages)) {
            return response.status(400).send({ error: 'Invalid messages format' });
        }

        if (!Object.values(PROMPT_PROCESSING_TYPE).includes(request.body.type)) {
            return response.status(400).send({ error: 'Unknown processing type' });
        }

        const messages = postProcessPrompt(request.body.messages, request.body.type, getPromptNames(request));
        return response.send({ messages });
    } catch (error) {
        console.error(error);
        return response.sendStatus(500);
    }
});
