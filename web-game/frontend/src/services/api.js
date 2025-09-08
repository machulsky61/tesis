import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

class GameAPI {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000, // 30 seconds timeout
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log('API Request:', config.method.toUpperCase(), config.url);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => {
        console.log('API Response:', response.status, response.config.url);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.status, error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async healthCheck() {
    const response = await this.client.get('/health');
    return response.data;
  }

  async detailedHealthCheck() {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }

  // Game operations
  async createGame(config) {
    const response = await this.client.post('/game/create', {
      config: config
    });
    return response.data;
  }

  async makeMove(gameId, pixel) {
    const response = await this.client.post(`/game/${gameId}/move`, {
      pixel: pixel
    });
    return response.data;
  }

  async getGameState(gameId) {
    const response = await this.client.get(`/game/${gameId}/state`);
    return response.data;
  }

  async getGameResult(gameId) {
    const response = await this.client.get(`/game/${gameId}/result`);
    return response.data;
  }

  async deleteGame(gameId) {
    const response = await this.client.delete(`/game/${gameId}`);
    return response.data;
  }

  // Utility endpoints
  async getAvailableJudges() {
    const response = await this.client.get('/game/judges');
    return response.data;
  }

  async getGameStats() {
    const response = await this.client.get('/game/stats');
    return response.data;
  }

  async cleanupOldGames(maxAgeHours = 1) {
    const response = await this.client.post('/game/cleanup', null, {
      params: { max_age_hours: maxAgeHours }
    });
    return response.data;
  }
}

// Create and export a singleton instance
const gameAPI = new GameAPI();
export default gameAPI;