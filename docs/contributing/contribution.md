This guide covers how to start contributing to Sedna 😄

## 1. Familiarize Yourself With Contributing to Sedna

### Setup GitHub Account

Sedna is developed on [GitHub][github] and will require
an account for contribution.

### Check The Kubernetes Contributor Guides

You may come back to this later, but we highly recommend reading these:

- [Kubernetes Contributor Guide](https://git.k8s.io/community/contributors/guide) 
  - Main contributor documentation, or you can just jump directly to the [contributing section](https://git.k8s.io/community/contributors/guide#contributing)
- [Contributor Cheat Sheet](https://git.k8s.io/community/contributors/guide/contributor-cheatsheet)
   - Common resources for existing developers


### Read The Sedna Docs 

These [proposals] cover framework and multi edge-cloud synergy capabilities of the project, which may be helpful to review before contributing. 

### Reaching Out

Issues are tracked on GitHub. Please check [the issue tracker][issues] to see
if there is any existing discussion or work related to your interests.

In particular, if you're just getting started, you may want to look for issues
labeled <a href="https://github.com/kubeedge/sedna/labels/good%20first%20issue" class="gh-label" style="background: #7057ff; color: white">good first issue</a> or <a href="https://github.com/kubeedge/sedna/labels/help%20wanted" class="gh-label" style="background: #006b75; color: white">help wanted</a> which are standard labels in the Kubernetes project.
The <a href="https://github.com/kubeedge/sedna/labels/help%20wanted" class="gh-label" style="background: #006b75; color: white">help wanted</a> label marks issues we're actively seeking help with while <a href="https://github.com/kubeedge/sedna/labels/good%20first%20issue" class="gh-label" style="background: #7057ff; color: white">good first issue</a> is additionally applied to a subset of issues we think will be particularly good for newcomers.

See [Kubernetes help wanted] for best practices.

If you're interested in working on any of these, leave a comment to let us know!

And please reach out in general for bugs, feature requests, and other issues!
If you do not see anything, please [file a new issue][file-an-issue].

> **NOTE**: _Please_ file an enhancement / [feature request issue][file-a-fr] to discuss features before filing a PR (ideally even before writing any code), we have a lot to consider with respect to our
> existing users and future support when accepting any new feature.
>
> To streamline the process, please reach out and discuss the concept and design
> / approach ASAP so the maintainers and community can get involved early.

## 2. Issue and pull request guidelines

- If you're looking to contribute documentation improvements, you'll specifically want to see the [kubernetes documentation style guide] before [filing an issue][file-an-issue].

- If you're planning to contribute code changes, you'll want to read the [development preparation guide] next.

- If you're planning to add a new synergy feature directly, you'll want to read the [guide][add-feature-guide] next.

When done, you can also refer our [recommended Git workflow] and [pull request best practices] before submitting a pull request.

## 3. Pull Request merging process
Pull requests are often called simply "PR".
Sedna generally follows the standard [github pull request](https://help.github.com/articles/about-pull-requests/) process. PR let you tell others about changes you've pushed to Sedna branch. Once a pull request is opened, you can discuss and review the potential changes with collaborators and add follow-up commits before your changes are merged into the main branch. 

When you have created and submitted a pull request (PR), we recommend the following steps to fasten its mergence.

- Attend the regular community meeting on a weekly basis. Pacific Time: **Thursday at 16:30-17:30 Beijing Time** (weekly, starting from Nov. 12th 2020).
([Convert to your timezone](https://www.thetimezoneconverter.com/?t=10%3A00&tz=GMT%2B8&))

- Propose a topic related to your PR to the meeting hosts, i.e., Zimu Zheng, Jie Pu and Siqi Luo, before each week's community meeting. You also can propose your topic by filling in [Meeting notes and agenda](https://docs.google.com/document/d/12n3kGUWTkAH4q2Wv5iCVGPTA_KRWav_eakbFrF9iAww/edit) before the meeting.

- In the community meeting, your PR will be discussed and reviewed publicly by SIG AI members.

- After PR is discussed and submitted for review, it can be merged.

### Code Review

To make it easier for your PR to receive reviews, consider the reviewers will need you to:

* follow [good coding guidelines](https://github.com/golang/go/wiki/CodeReviewComments).
* write [good commit messages](https://chris.beams.io/posts/git-commit/).
* break large changes into a logical series of smaller patches which individually make easily understandable changes, and in aggregate solve a broader issue.
* label PRs with appropriate reviewers: to do this read the messages the bot sends you to guide you through the PR process.

## 4. KubeEdge Community Membership

**Note :** This membership keeps changing based on the status and feedback of KubeEdge Community.

We gives a brief overview of the KubeEdge community roles with the requirements and responsibilities associated with them.

| Role | Requirements | Responsibilities | Privileges |
| -----| ---------------- | ------------ | -------|
| Member | Sponsor from 2 reviewers, active in community, multiple contributions to KubeEdge | Active contributor in the community | KubeEdge GitHub organization Member |
| Reviewer | Sponsor from 2 approvers, has good experience and history of review in specific package | Review contributions from other members | Add `lgtm` label to specific PRs |
| Approver | Sponsor from 2 maintainers, highly experienced and knowledge of domain, actively contributed to code and review  | Review and approve contributions from community members | Write access to specific packagies in relevant repository |
| Maintainer | Sponsor from 2 owners, shown good technical judgement in feature design/development and PR review | Participate in release planning and feature development/maintenance | Top level write access to relevant repository. Name entry in Maintainers file of the repository |
| Owner | Sponsor from 3 owners, helps drive the overall KubeEdge project | Drive the overall technical roadmap of the project and set priorities of activities in release planning | KubeEdge GitHub organization Admin access |


**Note :** It is mandatory for all KubeEdge community members to follow KubeEdge [Code of Conduct]. See [here](https://github.com/kubeedge/community/blob/master/community-membership.md) for more details.

[proposals]: /docs/proposals
[development preparation guide]: ./prepare-environment.md
[add-feature-guide]: control-plane/add-a-new-synergy-feature.md

[issues]: https://github.com/kubeedge/sedna/issues
[file-an-issue]: https://github.com/kubeedge/sedna/issues/new/choose
[file-a-fr]: https://github.com/kubeedge/sedna/issues/new?labels=kind%2Ffeature&template=enhancement.md

[github]: https://github.com/
[kubernetes documentation style guide]: https://github.com/kubernetes/community/blob/master/contributors/guide/style-guide.md
[recommended Git workflow]: https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md#workflow
[pull request best practices]: https://github.com/kubernetes/community/blob/master/contributors/guide/pull-requests.md#best-practices-for-faster-reviews
[Kubernetes help wanted]: https://www.kubernetes.dev/docs/guide/help-wanted/


