import { describe, expect, test } from '@jest/globals';

import { checkClaudeCachePrefix, compareClaudeCachePrompts, getClaudeCachePrompt } from '../src/endpoints/backends/claude-cache.js';

const text = value => ({ type: 'text', text: value });

/**
 * Builds the request of a chat with `turns` user turns.
 * @param {object} options
 * @param {number} options.turns Number of user turns
 * @param {string} [options.tail] Prompt placed after the chat history
 * @param {string} [options.note] Prompt injected four messages from the end
 * @param {boolean} [options.automatic] Use top-level automatic caching
 * @param {object} [options.extra] Extra request fields
 * @returns {object} Anthropic Messages request body
 */
function request({ turns, tail, note, automatic = true, extra = {} }) {
    const preset = [text('<additional instructions>'), text(`# Database ${'x'.repeat(4000)}`), text(`# Continuation Instructions ${'y'.repeat(4000)}`)];
    const messages = [{ role: 'user', content: [...preset, text(`user 1 ${'u'.repeat(300)}`)] }];
    for (let i = 2; i <= turns; i++) {
        messages.push({ role: 'assistant', content: [text(`assistant ${i - 1} ${'a'.repeat(300)}`)] });
        messages.push({ role: 'user', content: [text(`user ${i} ${'u'.repeat(300)}`)] });
    }
    if (tail) {
        messages.at(-1).content.push(text(tail));
    }
    if (note && messages.length > 4) {
        messages.at(-4).content.push(text(note));
    }
    return {
        model: 'claude-opus-5-5',
        system: [{ ...text('# System Instructions'), cache_control: { type: 'ephemeral', ttl: '1h' } }],
        messages,
        thinking: { type: 'adaptive' },
        ...(automatic ? { cache_control: { type: 'ephemeral', ttl: '1h' } } : {}),
        ...extra,
    };
}

const compare = (previous, current) => compareClaudeCachePrompts(getClaudeCachePrompt(previous), getClaudeCachePrompt(current));

describe('compareClaudeCachePrompts', () => {
    test('a chat that only grows reads the previous end of the prompt', () => {
        const report = compare(request({ turns: 5 }), request({ turns: 6 }));

        expect(report.hit).toBe(true);
        expect(report.lines[0]).toBe('previous prompt is intact (13 blocks, 2 new).');
        expect(report.lines.at(-1)).toMatch(/^expect a cache read up to messages\[8\]\.content\[0\] \(user\), about \d+% of this prompt\.$/);
    });

    test('an identical prompt reads everything', () => {
        expect(compare(request({ turns: 5 }), request({ turns: 5 })).hit).toBe(true);
    });

    test('a prompt after the chat history moves with every message', () => {
        const report = compare(request({ turns: 5, tail: '[Continue the chapter]' }), request({ turns: 6, tail: '[Continue the chapter]' }));

        expect(report.hit).toBe(false);
        expect(report.lines[0]).toMatch(/^prompt changed at block 14 of 16, messages\[9\]\.content\[0\] \(assistant\)/);
        expect(report.lines[1]).toBe('  was (messages[8].content[1] (user)): "[Continue the chapter]"');
        expect(report.lines.at(-1)).toMatch(/expect a full cache write/);
    });

    test('a prompt injected at a fixed depth moves with every message', () => {
        const report = compare(request({ turns: 5, note: '[Author note]' }), request({ turns: 6, note: '[Author note]' }));

        expect(report.hit).toBe(false);
        expect(report.lines[1]).toBe('  was (messages[5].content[1] (assistant)): "[Author note]"');
    });

    test('an edit inside a block is quoted around the edit', () => {
        const edited = request({ turns: 6 });
        edited.messages[0].content[1].text = `# Database ${'x'.repeat(2000)} 12:31 PM ${'x'.repeat(1990)}`;
        const report = compare(request({ turns: 5 }), edited);

        expect(report.hit).toBe(false);
        expect(report.lines[0]).toMatch(/messages\[0\]\.content\[1\] \(user\)/);
        expect(report.lines[2]).toMatch(/12:31 PM/);
    });

    test('a different model, thinking mode or effort has its own cache', () => {
        const report = compare(request({ turns: 5 }), request({ turns: 6, extra: { output_config: { effort: 'high' } } }));

        expect(report.hit).toBe(false);
        expect(report.lines[0]).toMatch(/effort changed/);
    });

    test('a breakpoint before a moving tail is still read', () => {
        const atDepth = (turns) => {
            const body = request({ turns, tail: '[Continue the chapter]', automatic: false });
            body.messages.at(-2).content.at(-1).cache_control = { type: 'ephemeral' };
            return body;
        };
        const report = compare(atDepth(5), atDepth(6));

        expect(report.hit).toBe(true);
        expect(report.lines.at(-1)).toMatch(/^expect a cache read up to messages\[7\]\.content\[0\] \(assistant\)/);
    });
});

describe('checkClaudeCachePrefix', () => {
    test('compares with the closest earlier request of the same user', () => {
        expect(checkClaudeCachePrefix('user', request({ turns: 5 }))).toBeNull();
        checkClaudeCachePrefix('user', { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'summarize' }], cache_control: { type: 'ephemeral' } });

        expect(checkClaudeCachePrefix('user', request({ turns: 6 })).hit).toBe(true);
        expect(checkClaudeCachePrefix('other', request({ turns: 6 }))).toBeNull();
    });
});
