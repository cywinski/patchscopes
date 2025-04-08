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

import ExplanationHead from '../heads/ExplanationHeadComponent';
import DreamHead from '../heads/DreamHeadComponent';
import TopWordsHead from '../heads/TopWordsHeadComponent';
import TopWordsBody from '../bodies/TopWordsBodyComponent';

/**
 * Providing a Card Component for the TopWords chart in different experiments.
 */
class TopWords extends React.PureComponent {
  /**
   * Rendering the chart card.
   *
   * @return {jsx} the component to be rendered.
   */
  render() {
    let iteration = this.props.dreamingElement.iteration ?
        this.props.dreamingElement.iteration : 0;
    iteration = iteration < this.props.dreamingElement.iterations.length ?
        iteration : this.props.dreamingElement.iterations.length - 1;
    const headParams = {
      'LayerID': this.props.dreamingElement.params.layer_id,
      'WordID': this.props.dreamingElement.params.word_id,
      'NeuronID': this.props.dreamingElement.params.neuron_id,
      'Activation': this.props.dreamingElement.iterations[
          iteration].activation.toFixed(4),
    };
    const target = [...this.props.dreamingElement.params.tokens];
    for (const tokenID in this.props.dreamingElement.params.tokens) {
      if (this.props.dreamingElement.word_id !== parseInt(tokenID)) {
        target[tokenID] = '';
      }
    }
    const sentenceParams = {
      headWidth: 30,
      colors: ['blue', 'black', 'black'],
      target: target,
    };
    const params = {
      tokens: this.props.dreamingElement.params.tokens,
    };
    const maxIterations = this.props.dreamingElement.iterations[
        this.props.dreamingElement.iterations.length -1].number;
    return (
      <Grid container direction='column' className='fullHeight' wrap='nowrap'>
        <ExplanationHead
          topic="Top Words"
          params={headParams}
          elementIndex={this.props.elementIndex}/>
        <DreamHead
          params={params}
          sentenceParams={sentenceParams}/>
        <TopWordsHead
          maxIterations={maxIterations}
          dreamingElement={this.props.dreamingElement}
          elementIndex={this.props.elementIndex}/>
        <Grid item xs>
          <Paper id='topWordsPaper' className={'dreamPaper fullHeight'}>
            <TopWordsBody
              dreamingElement={this.props.dreamingElement}
              elementIndex={this.props.elementIndex}/>
          </Paper>
        </Grid>
      </Grid>
    );
  }
}

TopWords.propTypes = {
  dreamingElement: PropTypes.object.isRequired,
  elementIndex: PropTypes.number.isRequired,
};

export default TopWords;
