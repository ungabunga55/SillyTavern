import fs from 'node:fs';
import path from 'node:path';

import express from 'express';
import _ from 'lodash';
import bytes from 'bytes';

import { SETTINGS_FILE } from '../constants.js';
import { getConfigValue, generateTimestamp, removeOldBackups, tryWriteFile } from '../util.js';
import { getAllUserHandles, getUserDirectories } from '../users.js';
import { getFileNameValidationFunction } from '../middleware/validateFileName.js';

const ENABLE_EXTENSIONS = !!getConfigValue('extensions.enabled', true, 'boolean');
const ENABLE_EXTENSIONS_AUTO_UPDATE = !!getConfigValue('extensions.autoUpdate', true, 'boolean');
const ENABLE_ACCOUNTS = !!getConfigValue('enableUserAccounts', false, 'boolean');
const ENABLE_REQUEST_COMPRESSION = !!getConfigValue('performance.requestCompression.enabled', false, 'boolean');
const REQUEST_COMPRESSION_MIN = bytes.parse(getConfigValue('performance.requestCompression.minPayloadSize', '256kb'));
const REQUEST_COMPRESSION_MAX = bytes.parse(getConfigValue('performance.requestCompression.maxPayloadSize', '8mb'));
const REQUEST_COMPRESSION_TIMEOUT = Number(getConfigValue('performance.requestCompression.timeout', 3000, 'number'));
const MAX_CONCURRENT_SETTINGS_READS = 8;

let activeSettingsReads = 0;
/** @type {Array<() => void>} */
const settingsReadWaiters = [];

async function readSettingsFile(filePath, encoding = null) {
    if (activeSettingsReads < MAX_CONCURRENT_SETTINGS_READS) {
        activeSettingsReads++;
    } else {
        await new Promise(resolve => settingsReadWaiters.push(resolve));
    }

    try {
        return await fs.promises.readFile(filePath, encoding ?? undefined);
    } finally {
        const next = settingsReadWaiters.shift();
        if (next) {
            next();
        } else {
            activeSettingsReads--;
        }
    }
}

async function mapSettingsFiles(files, callback) {
    const results = [];
    for (const batch of _.chunk(files, MAX_CONCURRENT_SETTINGS_READS)) {
        results.push(...await Promise.all(batch.map(callback)));
    }
    return results;
}

// 10 minutes
const AUTOSAVE_INTERVAL = 10 * 60 * 1000;

/**
 * Map of functions to trigger settings autosave for a user.
 * @type {Map<string, function>}
 */
const AUTOSAVE_FUNCTIONS = new Map();
/** @type {Set<Promise<void>>} */
const pendingSettingsBackups = new Set();

function runSettingsBackup(handle, preventDuplicates) {
    const backup = backupUserSettings(handle, preventDuplicates);
    pendingSettingsBackups.add(backup);
    const cleanup = () => pendingSettingsBackups.delete(backup);
    backup.then(cleanup, cleanup);
    return backup;
}

/**
 * Triggers autosave for a user every 10 minutes.
 * @param {string} handle User handle
 * @returns {void}
 */
function triggerAutoSave(handle) {
    if (!AUTOSAVE_FUNCTIONS.has(handle)) {
        const throttledAutoSave = _.throttle(() => {
            void runSettingsBackup(handle, true).catch(error => console.error(`Could not backup settings for ${handle}`, error));
        }, AUTOSAVE_INTERVAL);
        AUTOSAVE_FUNCTIONS.set(handle, throttledAutoSave);
    }

    const functionToCall = AUTOSAVE_FUNCTIONS.get(handle);
    if (functionToCall && typeof functionToCall === 'function') {
        functionToCall();
    }
}

export async function flushSettingsBackups() {
    for (const func of AUTOSAVE_FUNCTIONS.values()) {
        func.flush();
    }
    await Promise.allSettled([...pendingSettingsBackups]);
}

/**
 * Reads and parses files from a directory.
 * @param {string} directoryPath Path to the directory
 * @param {string} fileExtension File extension
 * @returns {Promise<Array>} Parsed files
 */
async function readAndParseFromDirectory(directoryPath, fileExtension = '.json') {
    const files = (await fs.promises.readdir(directoryPath))
        .filter(x => path.parse(x).ext == fileExtension)
        .sort();

    const results = await mapSettingsFiles(files, async item => {
        try {
            const file = await readSettingsFile(path.join(directoryPath, item), 'utf8');
            return { ok: true, value: fileExtension == '.json' ? JSON.parse(file) : file };
        } catch (error) {
            console.warn(`Could not read settings file ${path.join(directoryPath, item)}:`, error.message);
            return { ok: false };
        }
    });

    return results.filter(result => result.ok).map(result => result.value);
}

/**
 * Gets a sort function for sorting strings.
 * @param {*} _
 * @returns {(a: string, b: string) => number} Sort function
 */
function sortByName(_) {
    return (a, b) => a.localeCompare(b);
}

/**
 * Gets backup file prefix for user settings.
 * @param {string} handle User handle
 * @returns {string} File prefix
 */
export function getSettingsBackupFilePrefix(handle) {
    return `settings_${handle}_`;
}

async function readPresetsFromDirectory(directoryPath, options = {}) {
    const {
        sortFunction,
        removeFileExtension = false,
        fileExtension = '.json',
    } = options;

    const files = (await fs.promises.readdir(directoryPath)).sort(sortFunction).filter(x => path.parse(x).ext == fileExtension);

    const results = await mapSettingsFiles(files, async item => {
        try {
            const file = await readSettingsFile(path.join(directoryPath, item), 'utf8');
            JSON.parse(file);
            return { content: file, name: removeFileExtension ? item.replace(/\.[^/.]+$/, '') : item };
        } catch (error) {
            console.warn(`${item} is not a valid JSON:`, error.message);
            return null;
        }
    });

    const validResults = results.filter(Boolean);
    const fileContents = validResults.map(result => result.content);
    const fileNames = validResults.map(result => result.name);

    return { fileContents, fileNames };
}

async function backupSettings() {
    try {
        const userHandles = await getAllUserHandles();

        for (const handle of userHandles) {
            await runSettingsBackup(handle, true);
        }
    } catch (err) {
        console.error('Could not backup settings file', err);
    }
}

/**
 * Makes a backup of the user's settings file.
 * @param {string} handle User handle
 * @param {boolean} preventDuplicates Prevent duplicate backups
 * @returns {Promise<void>}
 */
async function backupUserSettings(handle, preventDuplicates) {
    const userDirectories = getUserDirectories(handle);

    if (!fs.existsSync(userDirectories.root)) {
        return;
    }

    const backupFile = path.join(userDirectories.backups, `${getSettingsBackupFilePrefix(handle)}${generateTimestamp()}.json`);
    const sourceFile = path.join(userDirectories.root, SETTINGS_FILE);

    if (!fs.existsSync(sourceFile)) {
        return;
    }

    const sourceContent = await readSettingsFile(sourceFile);
    if (preventDuplicates && await isDuplicateBackup(handle, sourceContent)) {
        return;
    }

    await tryWriteFile(backupFile, sourceContent);
    removeOldBackups(userDirectories.backups, getSettingsBackupFilePrefix(handle));
}

/**
 * Checks if the backup would be a duplicate.
 * @param {string} handle User handle
 * @param {Buffer} sourceContent Source file contents
 * @returns {Promise<boolean>} True if the backup is a duplicate
 */
async function isDuplicateBackup(handle, sourceContent) {
    const latestBackup = await getLatestBackup(handle);
    if (!latestBackup) {
        return false;
    }
    const latestContent = await readSettingsFile(latestBackup);
    return latestContent.equals(sourceContent);
}

/**
 * Gets the latest backup file for a user.
 * @param {string} handle User handle
 * @returns {Promise<string|null>} Latest backup file. Null if no backup exists.
 */
async function getLatestBackup(handle) {
    const userDirectories = getUserDirectories(handle);
    const prefix = getSettingsBackupFilePrefix(handle);
    const backupFiles = (await fs.promises.readdir(userDirectories.backups))
        .filter(file => file.startsWith(prefix) && /_\d{8}-\d{6}\.json$/.test(file))
        .sort((a, b) => b.localeCompare(a));
    const latestBackup = backupFiles[0];
    if (!latestBackup) {
        return null;
    }
    return path.join(userDirectories.backups, latestBackup);
}

export const router = express.Router();

router.post('/save', async function (request, response) {
    try {
        const pathToSettings = path.join(request.user.directories.root, SETTINGS_FILE);
        await tryWriteFile(pathToSettings, JSON.stringify(request.body, null, 4));
        response.send({ result: 'ok' });
        triggerAutoSave(request.user.profile.handle);
    } catch (err) {
        console.error(err);
        response.status(500).send({ error: 'Could not save settings.' });
    }
});

// Wintermute's code
router.post('/get', async (request, response) => {
    let settings;
    let directoryData;
    try {
        const pathToSettings = path.join(request.user.directories.root, SETTINGS_FILE);
        [settings, directoryData] = await Promise.all([
            readSettingsFile(pathToSettings, 'utf8'),
            Promise.all([
                readPresetsFromDirectory(request.user.directories.novelAI_Settings, {
                    sortFunction: sortByName(request.user.directories.novelAI_Settings),
                    removeFileExtension: true,
                }),
                readPresetsFromDirectory(request.user.directories.openAI_Settings, {
                    sortFunction: sortByName(request.user.directories.openAI_Settings), removeFileExtension: true,
                }),
                readPresetsFromDirectory(request.user.directories.textGen_Settings, {
                    sortFunction: sortByName(request.user.directories.textGen_Settings), removeFileExtension: true,
                }),
                readPresetsFromDirectory(request.user.directories.koboldAI_Settings, {
                    sortFunction: sortByName(request.user.directories.koboldAI_Settings), removeFileExtension: true,
                }),
                fs.promises.readdir(request.user.directories.worlds),
                readAndParseFromDirectory(request.user.directories.themes),
                readAndParseFromDirectory(request.user.directories.movingUI),
                readAndParseFromDirectory(request.user.directories.quickreplies),
                readAndParseFromDirectory(request.user.directories.instruct),
                readAndParseFromDirectory(request.user.directories.context),
                readAndParseFromDirectory(request.user.directories.sysprompt),
                readAndParseFromDirectory(request.user.directories.reasoning),
            ]),
        ]);
    } catch (e) {
        console.error('Could not load settings:', e);
        return response.sendStatus(500);
    }

    const [
        { fileContents: novelai_settings, fileNames: novelai_setting_names },
        { fileContents: openai_settings, fileNames: openai_setting_names },
        { fileContents: textgenerationwebui_presets, fileNames: textgenerationwebui_preset_names },
        { fileContents: koboldai_settings, fileNames: koboldai_setting_names },
        worldFiles,
        themes,
        movingUIPresets,
        quickReplyPresets,
        instruct,
        context,
        sysprompt,
        reasoning,
    ] = directoryData;

    const world_names = worldFiles
        .filter(file => path.extname(file).toLowerCase() === '.json')
        .sort((a, b) => a.localeCompare(b))
        .map(item => path.parse(item).name);

    response.send({
        settings,
        koboldai_settings,
        koboldai_setting_names,
        world_names,
        novelai_settings,
        novelai_setting_names,
        openai_settings,
        openai_setting_names,
        textgenerationwebui_presets,
        textgenerationwebui_preset_names,
        themes,
        movingUIPresets,
        quickReplyPresets,
        instruct,
        context,
        sysprompt,
        reasoning,
        enable_extensions: ENABLE_EXTENSIONS,
        enable_extensions_auto_update: ENABLE_EXTENSIONS_AUTO_UPDATE,
        enable_accounts: ENABLE_ACCOUNTS,
        request_compression: {
            enabled: ENABLE_REQUEST_COMPRESSION,
            minPayloadSize: REQUEST_COMPRESSION_MIN || 0,
            maxPayloadSize: REQUEST_COMPRESSION_MAX || 0,
            timeout: REQUEST_COMPRESSION_TIMEOUT || 0,
        },
    });
});

router.post('/get-snapshots', async (request, response) => {
    try {
        const snapshots = fs.readdirSync(request.user.directories.backups);
        const userFilesPattern = getSettingsBackupFilePrefix(request.user.profile.handle);
        const userSnapshots = snapshots.filter(x => x.startsWith(userFilesPattern));

        const result = userSnapshots.map(x => {
            const stat = fs.statSync(path.join(request.user.directories.backups, x));
            return { date: stat.ctimeMs, name: x, size: stat.size };
        });

        response.json(result);
    } catch (error) {
        console.error(error);
        response.sendStatus(500);
    }
});

router.post('/load-snapshot', getFileNameValidationFunction('name'), async (request, response) => {
    try {
        const userFilesPattern = getSettingsBackupFilePrefix(request.user.profile.handle);

        if (!request.body.name || !request.body.name.startsWith(userFilesPattern)) {
            return response.status(400).send({ error: 'Invalid snapshot name' });
        }

        const snapshotName = request.body.name;
        const snapshotPath = path.join(request.user.directories.backups, snapshotName);

        if (!fs.existsSync(snapshotPath)) {
            return response.sendStatus(404);
        }

        const content = await readSettingsFile(snapshotPath, 'utf8');

        response.send(content);
    } catch (error) {
        console.error(error);
        response.sendStatus(500);
    }
});

router.post('/make-snapshot', async (request, response) => {
    try {
        await runSettingsBackup(request.user.profile.handle, false);
        response.sendStatus(204);
    } catch (error) {
        console.error(error);
        response.sendStatus(500);
    }
});

router.post('/restore-snapshot', getFileNameValidationFunction('name'), async (request, response) => {
    try {
        const userFilesPattern = getSettingsBackupFilePrefix(request.user.profile.handle);

        if (!request.body.name || !request.body.name.startsWith(userFilesPattern)) {
            return response.status(400).send({ error: 'Invalid snapshot name' });
        }

        const snapshotName = request.body.name;
        const snapshotPath = path.join(request.user.directories.backups, snapshotName);

        if (!fs.existsSync(snapshotPath)) {
            return response.sendStatus(404);
        }

        const pathToSettings = path.join(request.user.directories.root, SETTINGS_FILE);
        const snapshotContent = await readSettingsFile(snapshotPath);
        await tryWriteFile(pathToSettings, snapshotContent);

        response.sendStatus(204);
    } catch (error) {
        console.error(error);
        response.sendStatus(500);
    }
});

/**
 * Initializes the settings endpoint
 */
export async function init() {
    await backupSettings();
}
