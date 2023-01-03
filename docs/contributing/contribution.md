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

## 2. What's Next?

- If you're looking to contribute documentation improvements, you'll specifically want to see the [kubernetes documentation style guide] before [filing an issue][file-an-issue].

- If you're planning to contribute code changes, you'll want to read the [development preparation guide] next.

- If you're planning to add a new synergy feature directly, you'll want to read the [guide][add-feature-guide] next.

When done, you can also refer our [recommended Git workflow] and [pull request best practices] before submitting a pull request.

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

## 3. Procedure to handle pull requests in Sedna
When you have created a pull request (PR), we recommend the following steps to fasten its mergence.

- Attend the regular community meeting on a weekly basis. Pacific Time: **Thursday at 16:30-17:30 Beijing Time** (weekly, starting from Nov. 12th 2020).
([Convert to your timezone](https://www.thetimezoneconverter.com/?t=10%3A00&tz=GMT%2B8&))

- Propose a topic related to your PR to the meeting hosts, i.e., Zimu Zheng, Jie Pu and Siqi Luo, before each week's community meeting. You also can propose your topic by filling in [Meeting notes and agenda](https://docs.google.com/document/d/12n3kGUWTkAH4q2Wv5iCVGPTA_KRWav_eakbFrF9iAww/edit) before the meeting.

- In the community meeting, your PR will be discussed and reviewed publicly by SIG AI members.

- After PR is discussed and submitted for review, it can be merged.

## 4. Contribution methods of different roles
In the community, there are a variety of roles that members play. Different roles focus on different ways to contribute. For example, developers will almost choose to contribute by codes while researchers might organize academic conference or share lectures to expand community influence.
