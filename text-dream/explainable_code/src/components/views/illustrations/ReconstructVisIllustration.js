import React from "react";
import { Grid } from "@material-ui/core";
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { bindActionCreators } from "redux";

import { getCard } from "../../../cardprocessing";
import * as actions from "../../../actions";
import { getReconstructExperimentExplanation } from "../../../data/ExperimentExplanationTexts";

/**
 * Displaying the illustration for this step in the explainable.
 */
class ReconstructVisIllustration extends React.Component {
  /**
   * Updating the page progress.
   */
  componentDidMount() {
    this.props.actions.loadReconstruction(this.props.dreamID);
    const cardElement = document.getElementById("cardItem");
    if (cardElement != null) {
      this.props.actions.changeCardDimensions({
        width: cardElement.getBoundingClientRect().width,
        height: cardElement.getBoundingClientRect().height,
      });
    }
  }

  /**
   * Renders the illustration.
   *
   * @return {jsx} the component to be rendered.
   */
  render() {
    const reconstructCard = getCard(
      this.props.reconstructVisJSON,
      0,
      getReconstructExperimentExplanation(this.props.dreamID),
      ["temperature", "loss", "ids_loss"]
    );
    return (
      <Grid item xs className="fullHeight" id="cardItem">
        {reconstructCard}
      </Grid>
    );
  }
}

ReconstructVisIllustration.propTypes = {
  actions: PropTypes.object.isRequired,
  dreamID: PropTypes.number.isRequired,
  reconstructVisJSON: PropTypes.object.isRequired,
};

/**
 * Mapping the state that this component needs to its props.
 *
 * @param {object} state - the application state from where to get needed props.
 * @param {object} ownProps - optional own properties needed to acess state.
 * @return {object} the props for this component.
 */
function mapStateToProps(state, ownProps) {
  return {
    dreamID: state.dreamID,
    reconstructVisJSON: state.reconstructVisJSON,
  };
}

/**
 * Mapping the actions of redux to this component.
 *
 * @param {function} dispatch - called whenever an action is to be dispatched.
 * @return {object} all the actions bound to this component.
 */
function mapDispatchToProps(dispatch) {
  return { actions: bindActionCreators(actions, dispatch) };
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(ReconstructVisIllustration);
