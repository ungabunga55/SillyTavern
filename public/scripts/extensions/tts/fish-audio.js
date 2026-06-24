import { event_types, eventSource, getRequestHeaders } from '../../../script.js';
import { SECRET_KEYS, secret_state } from '../../secrets.js';
import { getPreviewString, saveTtsProviderSettings } from './index.js';

export { FishAudioTtsProvider };

class FishAudioTtsProvider {
    settings;
    voices = [];
    separator = ' . ';
    audioElement = document.createElement('audio');

    defaultSettings = {
        voiceMap: {},
        model: 's2.1-pro-free',
        voiceId: '',
        temperature: 0.7,
        top_p: 0.7,
    };

    static models = [
        { id: 's2.1-pro-free', name: 's2.1-pro-free' },
        { id: 's2-pro', name: 's2-pro' },
        { id: 's1', name: 's1' },
    ];

    get settingsHtml() {
        return `
        <div>Fish Audio TTS API.</div>
        <div class="flex-container alignItemsCenter">
            <div class="flex1"></div>
            <div id="fish_audio_tts_key" class="menu_button menu_button_icon manage-api-keys" data-key="api_key_fish_audio">
                <i class="fa-solid fa-key"></i>
                <span>API Key</span>
            </div>
        </div>
        <div class="flex-container flexFlowColumn">
            <div>
                <label for="fish_audio_tts_model">Model</label>
                <select id="fish_audio_tts_model" class="text_pole">
                    ${FishAudioTtsProvider.models.map(model => `<option value="${model.id}">${model.name}</option>`).join('')}
                </select>
            </div>
            <div>
                <label for="fish_audio_tts_voice_id">Voice ID</label>
                <input id="fish_audio_tts_voice_id" class="text_pole" type="text" placeholder="Fish Audio reference_id / model ID" />
            </div>
            <div>
                <label for="fish_audio_tts_temperature">Temperature <span id="fish_audio_tts_temperature_output"></span></label>
                <input id="fish_audio_tts_temperature" type="range" value="0.7" min="0" max="1" step="0.05" />
                <small>Controls expressiveness. Higher is more varied, lower is more consistent.</small>
            </div>
            <div>
                <label for="fish_audio_tts_top_p">Top P <span id="fish_audio_tts_top_p_output"></span></label>
                <input id="fish_audio_tts_top_p" type="range" value="0.7" min="0" max="1" step="0.05" />
                <small>Controls diversity via nucleus sampling.</small>
            </div>
        </div>`;
    }

    constructor() {
        this.handler = async function (/** @type {string} */ key) {
            if (key !== SECRET_KEYS.FISH_AUDIO) return;
            $('#fish_audio_tts_key').toggleClass('success', !!secret_state[SECRET_KEYS.FISH_AUDIO]);
        }.bind(this);
    }

    dispose() {
        [event_types.SECRET_WRITTEN, event_types.SECRET_DELETED, event_types.SECRET_ROTATED].forEach(event => {
            eventSource.removeListener(event, this.handler);
        });
    }

    async loadSettings(settings) {
        this.settings = { ...this.defaultSettings, ...settings };
        this.settings.voiceMap ??= {};

        $('#fish_audio_tts_model').val(this.settings.model);
        $('#fish_audio_tts_voice_id').val(this.settings.voiceId);
        $('#fish_audio_tts_temperature').val(this.settings.temperature);
        $('#fish_audio_tts_temperature_output').text(this.settings.temperature);
        $('#fish_audio_tts_top_p').val(this.settings.top_p);
        $('#fish_audio_tts_top_p_output').text(this.settings.top_p);

        $('#fish_audio_tts_key').toggleClass('success', !!secret_state[SECRET_KEYS.FISH_AUDIO]);
        [event_types.SECRET_WRITTEN, event_types.SECRET_DELETED, event_types.SECRET_ROTATED].forEach(event => {
            eventSource.on(event, this.handler);
        });

        $('#fish_audio_tts_model').on('change', () => this.onSettingsChange());
        $('#fish_audio_tts_voice_id').on('input', () => this.onSettingsChange());
        $('#fish_audio_tts_temperature').on('input', () => this.onSettingsChange());
        $('#fish_audio_tts_top_p').on('input', () => this.onSettingsChange());

        await this.checkReady();
    }

    onSettingsChange() {
        this.settings.model = String($('#fish_audio_tts_model').val() || this.defaultSettings.model);
        this.settings.voiceId = String($('#fish_audio_tts_voice_id').val() || '').trim();
        this.settings.temperature = Number($('#fish_audio_tts_temperature').val());
        this.settings.top_p = Number($('#fish_audio_tts_top_p').val());
        $('#fish_audio_tts_temperature_output').text(this.settings.temperature);
        $('#fish_audio_tts_top_p_output').text(this.settings.top_p);
        saveTtsProviderSettings();
    }

    async checkReady() {
        this.voices = await this.fetchTtsVoiceObjects();
    }

    async onRefreshClick() {
        return this.checkReady();
    }

    getConfiguredVoice() {
        return {
            name: 'Configured Voice',
            voice_id: this.settings.voiceId,
            lang: '',
            preview_url: false,
        };
    }

    async getVoice(voiceName) {
        const voice = this.getConfiguredVoice();

        if (!voiceName || voiceName === voice.name || voiceName === voice.voice_id) {
            return voice;
        }

        throw new Error(`TTS Voice not found: ${voiceName}`);
    }

    async generateTts(text, voiceId) {
        return this.fetchTtsGeneration(text, voiceId);
    }

    async fetchTtsVoiceObjects() {
        return [this.getConfiguredVoice()];
    }

    async previewTtsVoice(voiceId) {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;

        const response = await this.fetchTtsGeneration(getPreviewString('en-US'), voiceId || this.settings.voiceId);
        const audio = await response.blob();
        const url = URL.createObjectURL(audio);
        this.audioElement.src = url;
        this.audioElement.play();
        this.audioElement.onended = () => URL.revokeObjectURL(url);
    }

    async fetchTtsGeneration(text, voiceId) {
        if (!secret_state[SECRET_KEYS.FISH_AUDIO]) {
            throw new Error('No Fish Audio API key found');
        }

        const referenceId = String(voiceId || this.settings.voiceId || '').trim();
        if (!referenceId) {
            throw new Error('Fish Audio voice ID is not set');
        }

        const response = await fetch('/api/openai/fish-audio/generate-voice', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                input: text,
                voice: referenceId,
                model: this.settings.model,
                temperature: this.settings.temperature,
                top_p: this.settings.top_p,
            }),
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Fish Audio TTS failed: ${error}`);
        }

        return response;
    }
}
