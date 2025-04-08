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
import React from 'react';
import PropTypes from 'prop-types';

import {Grid, Paper} from '@material-ui/core';

import DreamBody from '../bodies/DreamBodyComponent';
import DreamHead from '../heads/DreamHeadComponent';
import ExplanationHead from '../heads/ExplanationHeadComponent';

import * as sentences from '../../sentences';

/**
 * Provides the Dream Card Component.
 */
class Dream extends React.PureComponent {
  /**
   * Renders the Dream Card.
   *
   * @return {jsx} the dream card to be rendered
   */
  render() {
    const sentenceParams = sentences.getDreamSentenceParams(
        this.props.results, this.props.params);
    const headParams = {
      'LayerID': this.props.params.layer_id,
      'WordID': this.props.params.word_id,
      'NeuronID': this.props.params.neuron_id,
    };
    return (
      <Grid container direction='column' className='fullHeight' wrap='nowrap'>
        <ExplanationHead
          topic="Dream"
          params={headParams}
          elementIndex={this.props.elementIndex}/>
        <DreamHead
          params={this.props.params}
          sentenceParams={sentenceParams}/>
        <div className='overflow'>
          <Paper className={'dreamPaper'}>
            <DreamBody
              results={this.props.results}
              params={this.props.params}
              sentenceParams={sentenceParams}/>
          </Paper>
        </div>
      </Grid>
    );
  }
}

Dream.propTypes = {
  params: PropTypes.object.isRequired,
  results: PropTypes.object.isRequired,
  elementIndex: PropTypes.number.isRequired,
};

export default Dream;
