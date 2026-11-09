/**
 * Claude reads the prompt cache by exact prefix: an entry written at a breakpoint is only
 * usable while everything before that breakpoint is sent again unchanged. This compares each
 * request with the ones before it and reports where the prompt stopped matching.
 */

/** Positions a breakpoint looks back over for an earlier entry. */
const LOOKBACK_POSITIONS = 20;
/** Requests remembered per user, so other chats and background prompts do not hide the match. */
const HISTORY_SIZE = 8;
const SNIPPET_LENGTH = 70;
/** Prefixes shorter than about 512 tokens are never cached. */
const MIN_CACHEABLE_LENGTH = 1500;

/** @type {Map<string, CachePrompt[]>} */
const history = new Map();

/**
 * @typedef {object} CacheBlock
 * @property {string} path Where the block sits in the request
 * @property {string} identity Everything about the block that is part of the prompt
 * @property {string} text Readable content, for reporting
 *
 * @typedef {object} CachePrompt
 * @property {string} settings Request settings that select a separate cache
 * @property {CacheBlock[]} blocks Blocks in cache order: tools, system, messages
 * @property {number[]} breakpoints Indexes of the blocks that write a cache entry
 */

/**
 * Lists a request's blocks in the order Claude caches them.
 * @param {object} body Anthropic Messages request body
 * @returns {CachePrompt}
 */
export function getClaudeCachePrompt(body) {
    /** @type {CacheBlock[]} */
    const blocks = [];
    /** @type {number[]} */
    const breakpoints = [];
    const add = (path, block, role) => {
        const { cache_control: marker, ...content } = block ?? {};
        if (marker) {
            breakpoints.push(blocks.length);
        }
        const text = typeof content.text === 'string' ? content.text : JSON.stringify(content);
        blocks.push({ path, identity: JSON.stringify([role, content]), text });
    };

    (Array.isArray(body.tools) ? body.tools : []).forEach((tool, i) => add(`tools[${i}]`, tool));
    (Array.isArray(body.system) ? body.system : []).forEach((block, i) => add(`system[${i}]`, block));
    (Array.isArray(body.messages) ? body.messages : []).forEach((message, i) => {
        const content = Array.isArray(message?.content) ? message.content : [{ type: 'text', text: message?.content }];
        content.forEach((block, j) => add(`messages[${i}].content[${j}] (${message?.role})`, block, message?.role));
    });

    // Automatic caching writes at the last block that can carry a breakpoint.
    if (body.cache_control) {
        const last = blocks.findLastIndex(block => !/^\[[^,]*,\{"type":"(redacted_)?thinking"/.test(block.identity));
        if (last >= 0 && !breakpoints.includes(last)) {
            breakpoints.push(last);
        }
    }

    const settings = JSON.stringify([body.model, body.thinking?.type ?? null, body.output_config?.effort ?? null]);
    return { settings, blocks, breakpoints: breakpoints.sort((a, b) => a - b) };
}

/**
 * Shortens text around the first character that differs from another text.
 * @param {string} text Text to quote
 * @param {string} other Text it is compared with
 * @returns {string}
 */
function quoteDifference(text, other) {
    let start = 0;
    while (start < text.length && text[start] === other[start]) {
        start++;
    }
    const from = Math.max(0, start - 20);
    const snippet = text.slice(from, from + SNIPPET_LENGTH).replace(/\s+/g, ' ');
    return `${from > 0 ? '…' : ''}${snippet}${from + SNIPPET_LENGTH < text.length ? '…' : ''}`;
}

/**
 * Explains what an earlier request left in the cache that this request can read.
 * @param {CachePrompt} previous Earlier request
 * @param {CachePrompt} current Request about to be sent
 * @returns {{hit: boolean, common: number, lines: string[]}} `hit` when every entry it wrote is still readable
 */
export function compareClaudeCachePrompts(previous, current) {
    if (previous.settings !== current.settings) {
        return { hit: false, common: 0, lines: [`model, thinking mode or effort changed (${previous.settings} -> ${current.settings}); each combination has its own cache.`] };
    }

    let common = 0;
    while (common < previous.blocks.length && common < current.blocks.length && previous.blocks[common].identity === current.blocks[common].identity) {
        common++;
    }

    const size = count => current.blocks.slice(0, count).reduce((sum, block) => sum + block.text.length, 0);
    const share = count => Math.round(100 * size(count) / Math.max(1, size(current.blocks.length)));
    const readable = previous.breakpoints.filter(index => index < common && size(index + 1) >= MIN_CACHEABLE_LENGTH).at(-1);
    const inReach = readable !== undefined && current.breakpoints.some(index => index >= readable && index - readable <= LOOKBACK_POSITIONS);
    const lines = [];

    if (common === previous.blocks.length) {
        lines.push(`previous prompt is intact (${common} blocks, ${current.blocks.length - common} new).`);
    } else {
        const was = previous.blocks[common];
        const now = current.blocks[common];
        lines.push(`prompt changed at block ${common + 1} of ${current.blocks.length}, ${now?.path ?? 'the end of the prompt'}; ${share(common)}% of the prompt before it is unchanged.`);
        lines.push(`  was (${was.path}): "${quoteDifference(was.text, now?.text ?? '')}"`);
        lines.push(`  now: ${now ? `"${quoteDifference(now.text, was.text)}"` : 'nothing, the prompt ends here'}`);
    }

    const lost = previous.breakpoints.filter(index => index >= common).map(index => previous.blocks[index].path);
    if (lost.length) {
        lines.push(`cache entries written after the change cannot be read: ${lost.join(', ')}.`);
    }
    if (readable === undefined) {
        lines.push('nothing cacheable was written before the change: expect a full cache write.');
    } else if (!inReach) {
        lines.push(`the entry at ${previous.blocks[readable].path} is more than ${LOOKBACK_POSITIONS} blocks before any breakpoint of this request: expect a full cache write.`);
    } else {
        lines.push(`expect a cache read up to ${previous.blocks[readable].path}, about ${share(readable + 1)}% of this prompt.`);
    }

    return { hit: inReach && lost.length === 0, common, lines };
}

/**
 * Compares a request with the user's recent ones and reports what it can read from the cache.
 * @param {string} user Handle of the requesting user
 * @param {object} body Anthropic Messages request body
 * @returns {{hit: boolean, lines: string[]}|null} Report against the closest earlier request, if any
 */
export function checkClaudeCachePrefix(user, body) {
    const current = getClaudeCachePrompt(body);
    const recent = history.get(user) ?? [];
    const reports = recent.map(previous => compareClaudeCachePrompts(previous, current));
    const best = reports.reduce((a, b) => (!a || b.common >= a.common ? b : a), null);

    history.set(user, [...recent, current].slice(-HISTORY_SIZE));
    return best;
}
