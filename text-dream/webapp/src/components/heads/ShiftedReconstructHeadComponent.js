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

import {Grid, Typography, Tooltip, Paper} from '@material-ui/core';

import ReconstructSentence from '../reconstruct/ReconstructSentence';

/**
 * Providing a header component for shifted Reconstruct results.
 */
class ShiftedReconstructHead extends React.Component {
  /**
   * Render the header component.
   *
   * @return {jsx} the component to be rendered.
   */
  render() {
    return (
      <Grid item>
        <Paper className='subHeadingPaper' style={{backgroundColor: '#DDDDDD'}}
          square>
          <Grid container direction='row' spacing={1} alignItems="center">
            <Tooltip title="Reconstruct Input" placement="top">
              <Grid item style={{width: this.props.sentenceParams.headWidth}}>
                <Typography variant="body1" color="inherit">
                  I
                </Typography>
              </Grid>
            </Tooltip>
            <Grid item>
              <ReconstructSentence
                sentence={this.props.params.tokens}
                target={this.props.sentenceParams.target}
                original={this.props.params.tokens}
                colors={this.props.sentenceParams.colors}/>
            </Grid>
          </Grid>
        </Paper>
        <Paper className='subHeadingPaper' style={{backgroundColor: '#DDDDDD'}}
          square>
          <Grid container direction='row' spacing={1} alignItems="center">
            <Tooltip title="Reconstruct Target" placement="top">
              <Grid item style={{width: this.props.sentenceParams.headWidth}}>
                <Typography variant="body1" color="inherit">
                  T
                </Typography>
              </Grid>
            </Tooltip>
            <Grid item>
              <ReconstructSentence
                sentence={this.props.sentenceParams.changedSentence}
                target={this.props.sentenceParams.target}
                original={this.props.params.tokens}
                colors={this.props.sentenceParams.colors}/>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    );
  }
}

ShiftedReconstructHead.propTypes = {
  params: PropTypes.object.isRequired,
  sentenceParams: PropTypes.object.isRequired,
};

export default ShiftedReconstructHead;
