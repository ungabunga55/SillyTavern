import fetch from 'node-fetch';

const MODELS_PER_PAGE = 1000;
const MAX_MODEL_PAGES = 100;

/**
 * Fetches the complete Featherless model catalog.
 * @param {string | URL} modelsEndpoint Featherless /v1/models endpoint
 * @param {Record<string, string>} headers Request headers
 * @returns {Promise<Record<string, any>[]>} Model records
 */
export async function fetchFeatherlessModels(modelsEndpoint, headers) {
    const models = [];
    const modelIds = new Set();

    for (let page = 1; page <= MAX_MODEL_PAGES; page++) {
        const url = new URL(modelsEndpoint);
        url.searchParams.set('page', String(page));
        url.searchParams.set('per_page', String(MODELS_PER_PAGE));

        const response = await fetch(url, { method: 'GET', headers });
        if (!response.ok) {
            const responseText = await response.text();
            const error = new Error(`Featherless models endpoint failed: ${response.status} ${response.statusText}`);
            error['status'] = response.status;
            error['responseText'] = responseText;
            throw error;
        }

        /** @type {any} */
        const data = await response.json();
        if (!Array.isArray(data?.data)) {
            throw new Error('Featherless models endpoint did not return a model list.');
        }

        let added = 0;
        for (const model of data.data) {
            const id = String(model?.id || '').trim();
            if (!id || modelIds.has(id)) {
                continue;
            }
            modelIds.add(id);
            models.push(model);
            added++;
        }

        if (data.data.length < MODELS_PER_PAGE) {
            break;
        }
        if (added === 0 || page === MAX_MODEL_PAGES) {
            throw new Error('Featherless model pagination did not reach the end of the catalog.');
        }
    }

    return models;
}
