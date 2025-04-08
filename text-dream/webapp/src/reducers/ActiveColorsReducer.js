/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
*/
import initialState from './initialState';
import * as types from '../actions/types';

/**
 * Reducer for updating the dimensions of displayed cards.
 *
 * @param {object} state - the current application state before any change.
 * @param {object} action - the action that is issued to manipulate the state.
 * @return {object} the state after handling the actiton.
 */
export default function activeColorsReducer(
    state = initialState.activeColors, action) {
  switch (action.type) {
    case types.ADD_ACTIVE_COLORS:
      return state.concat([action.colors]);
    case types.REMOVE_ACTIVE_COLORS: {
      const removedColorsState = [...state];
      removedColorsState.splice(action.index, 1);
      return removedColorsState;
    }
    default:
      return state;
  }
}
