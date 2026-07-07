import express from 'express';
import fetch from 'node-fetch';
import mime from 'mime-types';
import { readSecret, SECRET_KEYS } from './secrets.js';
import { getOpenRouterHeaders } from '../constants.js';

export const router = express.Router();
const API_OPENROUTER = 'https://openrouter.ai/api/v1';
const SUPPORTED_OPENROUTER_IMAGE_FORMATS = new Set(['png', 'jpg', 'jpeg', 'webp']);

router.post('/models/providers', async (req, res) => {
    try {
        const { model } = req.body;
        const response = await fetch(`${API_OPENROUTER}/models/${model}/endpoints`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
        });

        if (!response.ok) {
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();
        const endpoints = data?.data?.endpoints || [];
        const providerNames = endpoints.map(e => e.provider_name);

        return res.json(providerNames);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

/**
 * Fetches and filters models from OpenRouter API based on modality criteria.
 * @param {string} endpoint - The API endpoint to fetch from
 * @param {string} inputModality - Required input modality
 * @param {string} outputModality - Required output modality
 * @param {((model: any) => any) | null} [mapFn=null] - Optional mapping function to transform the results
 * @returns {Promise<any[]>} Filtered and/or mapped models
 */
async function fetchModelsByModality(endpoint, inputModality, outputModality, mapFn = null) {
    const response = await fetch(`${API_OPENROUTER}${endpoint}?output_modalities=${encodeURIComponent(outputModality)}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
        console.warn('OpenRouter API request failed', response.statusText);
        return [];
    }

    /** @type {any} */
    const data = await response.json();

    if (!Array.isArray(data?.data)) {
        console.warn('OpenRouter API response was not an array');
        return [];
    }

    const filtered = data.data
        .filter(m => Array.isArray(m?.architecture?.input_modalities))
        .filter(m => m.architecture.input_modalities.includes(inputModality))
        .filter(m => Array.isArray(m?.architecture?.output_modalities))
        .filter(m => m.architecture.output_modalities.includes(outputModality))
        .sort((a, b) => a?.id && b?.id ? a.id.localeCompare(b.id) : 0);

    return typeof mapFn === 'function' ? filtered.map(mapFn) : filtered;
}

router.post('/models/multimodal', async (_req, res) => {
    try {
        const models = await fetchModelsByModality('/models', 'image', 'text', m => m.id);
        return res.json(models);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.post('/models/embedding', async (_req, res) => {
    try {
        const models = await fetchModelsByModality('/models', 'text', 'embeddings', m => ({ id: m.id, name: m.name }));
        return res.json(models);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.post('/models/image', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.OPENROUTER);
        const headers = { 'Accept': 'application/json' };

        if (key) {
            headers['Authorization'] = `Bearer ${key}`;
        }

        const response = await fetch(`${API_OPENROUTER}/images/models`, {
            method: 'GET',
            headers,
        });

        if (!response.ok) {
            console.warn('OpenRouter image models request failed', response.statusText);
            return res.json([]);
        }

        /** @type {any} */
        const data = await response.json();

        if (!Array.isArray(data?.data)) {
            console.warn('OpenRouter image models response was not an array');
            return res.json([]);
        }

        const models = data.data
            .filter(m => Array.isArray(m?.architecture?.input_modalities))
            .filter(m => m.architecture.input_modalities.includes('text'))
            .filter(m => Array.isArray(m?.architecture?.output_modalities))
            .filter(m => m.architecture.output_modalities.includes('image'))
            .sort((a, b) => a?.id && b?.id ? a.id.localeCompare(b.id) : 0)
            .map(m => ({
                value: m.id,
                text: m.name || m.id,
                supported_parameters: m.supported_parameters || {},
                supports_streaming: !!m.supports_streaming,
            }));

        return res.json(models);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.post('/credits', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.OPENROUTER);

        if (!key) {
            console.warn('OpenRouter API key not found');
            return res.sendStatus(400);
        }

        const response = await fetch(`${API_OPENROUTER}/credits`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Authorization': `Bearer ${key}`,
            },
        });

        if (!response.ok) {
            console.warn('OpenRouter credits request failed', response.statusText);
            return res.sendStatus(500);
        }

        /** @type {any} */
        const data = await response.json();
        const totalCredits = data.data?.total_credits ?? 0;
        const totalUsage = data.data?.total_usage ?? 0;
        const remaining = totalCredits - totalUsage;

        return res.json({ remaining, total_credits: totalCredits, total_usage: totalUsage });
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});

router.post('/image/generate', async (req, res) => {
    try {
        const key = readSecret(req.user.directories, SECRET_KEYS.OPENROUTER);

        if (!key) {
            console.warn('OpenRouter API key not found');
            return res.status(400).json({ error: 'OpenRouter API key not found' });
        }

        console.debug('OpenRouter image generation request', req.body);

        const { model, prompt } = req.body;

        if (!model || !prompt) {
            return res.status(400).json({ error: 'Model and prompt are required' });
        }

        const requestBody = {
            model: model,
            prompt: prompt,
            n: 1,
        };

        if (req.body.aspect_ratio) {
            requestBody.aspect_ratio = req.body.aspect_ratio;
        }

        if (req.body.seed !== undefined) {
            requestBody.seed = req.body.seed;
        }

        const response = await fetch(`${API_OPENROUTER}/images`, {
            method: 'POST',
            headers: {
                ...getOpenRouterHeaders(req.body),
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${key}`,
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            console.warn('OpenRouter image generation failed', await response.text());
            return res.sendStatus(500);
        }

        /** @type {any} */
        const data = await response.json();

        const encodedImage = String(data?.data?.[0]?.b64_json || '');

        if (!encodedImage) {
            console.warn('No image data found in OpenRouter response', data);
            return res.sendStatus(500);
        }

        const dataUrlMatch = /^data:(.*);base64,(.*)$/.exec(encodedImage);
        const mediaType = data?.data?.[0]?.media_type;
        const mimeType = dataUrlMatch?.[1] || mediaType || 'image/png';
        const base64Data = dataUrlMatch?.[2] || encodedImage;
        const format = mime.extension(mimeType) || 'png';

        if (!base64Data) {
            console.warn('Invalid image data format', data);
            return res.sendStatus(500);
        }

        if (!SUPPORTED_OPENROUTER_IMAGE_FORMATS.has(format)) {
            console.warn('Unsupported OpenRouter image format', mimeType);
            return res.status(500).json({ error: `Unsupported image format: ${format}` });
        }

        const result = {
            format: format,
            image: base64Data,
        };

        return res.json(result);
    } catch (error) {
        console.error(error);
        return res.sendStatus(500);
    }
});
