#!/usr/bin/env python

# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# used to generate copyright from hack/boilerplate.
# shoud be executed in root dir:
#   hack/boilerplate/gen_boilerplate.py
# 
# modify from https://github.com/kubernetes/kubernetes/blob/master/hack/boilerplate/boilerplate.py
# which is used for check copyright

from __future__ import print_function

import argparse
import datetime
import glob
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "filenames",
    help="list of files to generate copyright, all files if unspecified",
    nargs='*')

rootdir = os.path.dirname(__file__) + "/../../"
rootdir = os.path.abspath(rootdir)
parser.add_argument(
    "--rootdir", default=rootdir, help="root directory to generate")

default_boilerplate_dir = os.path.join(rootdir, "hack/boilerplate")
parser.add_argument(
    "--boilerplate-dir", default=default_boilerplate_dir)

parser.add_argument(
    "-v", "--verbose",
    help="give verbose output regarding why a file does not pass",
    action="store_true")

args = parser.parse_args()

verbose_out = sys.stderr if args.verbose else open("/dev/null", "w")


def get_year():
    return str(datetime.datetime.now().year)

def get_refs():
    refs = {}

    for path in glob.glob(os.path.join(args.boilerplate_dir, "boilerplate.*.txt")):
        extension = os.path.basename(path).split(".")[1]

        ref_file = open(path, 'r')
        ref = ref_file.read()
        ref_file.close()
        refs[extension] = ref.replace('YEAR', get_year())

    return refs


def is_generated_file(filename, data, regexs):
    for d in skipped_ungenerated_files:
        if d in filename:
            return False

    p = regexs["generated"]
    return p.search(data)


def file_update(filename, refs, regexs):
    try:
        f = open(filename, 'r')
    except Exception as exc:
        print("Unable to open %s: %s" % (filename, exc), file=verbose_out)
        return False

    data = f.read()
    f.close()

    if 'Copyright' in data[:100]:
        print("%s already has copyright." % filename, file=verbose_out)
        return True
    if not data.strip():
        # if nothing, no need to add
        return True

    # determine if the file is automatically generated
    generated = is_generated_file(filename, data, regexs)

    basename = os.path.basename(filename)
    extension = file_extension(filename)
    if generated:
        if extension == "go":
            extension = "generatego"
        elif extension == "bzl":
            extension = "generatebzl"

    if extension != "":
        ref = refs[extension]
    else:
        ref = refs[basename]

    prefix = ''
    # remove extra content from the top of files
    if extension == "go" or extension == "generatego":
        p = regexs["go_build_constraints"]
        #(data, found) = p.subn("", data, 1)
        m = p.match(data)
        if m:
            prefix = m.group()
            data = data[len(prefix):]
    elif extension in ["sh", "py"]:
        p = regexs["shebang"]
        m = p.match(data)
        if m:
            prefix = m.group()
            data = data[len(prefix):]

    try:
        with open(filename, 'w') as f:
            f.write(prefix + ref + data)
    except Exception as exc:
        print("Unable to write %s: %s" % (filename, exc), file=verbose_out)
        return False
    return True


def file_extension(filename):
    return os.path.splitext(filename)[1].split(".")[-1].lower()


skipped_dirs = ['third_party', '_gopath', '_output', '.git',
                "vendor",
                ]

skipped_files = [
    'docs/conf.py', 'docs/Makefile']

skipped_ungenerated_files = [
    'hack/boilerplate/boilerplate.py']


def normalize_files(files):
    newfiles = []
    for pathname in files:
        if any(x in pathname for x in skipped_dirs):
            continue
        newfiles.append(pathname)
    for i, pathname in enumerate(newfiles):
        if not os.path.isabs(pathname):
            newfiles[i] = os.path.join(args.rootdir, pathname)
    return newfiles


def get_files(extensions):
    files = []
    if len(args.filenames) > 0:
        files = args.filenames
    else:
        for root, dirs, walkfiles in os.walk(args.rootdir):
            # don't visit certain dirs. This is just a performance improvement
            # as we would prune these later in normalize_files(). But doing it
            # cuts down the amount of filesystem walking we do and cuts down
            # the size of the file list
            for d in skipped_dirs:
                if d in dirs:
                    dirs.remove(d)

            for name in walkfiles:
                pathname = os.path.join(root, name)
                files.append(pathname)

    files = normalize_files(files)
    outfiles = []
    skipped_norm_files = normalize_files(skipped_files)

    for pathname in files:
        if pathname in skipped_norm_files:
            continue
        basename = os.path.basename(pathname)
        extension = file_extension(pathname)
        if extension in extensions or basename in extensions:
            outfiles.append(pathname)
    return outfiles


def get_regexs():
    regexs = {}
    # strip // +build \n\n build constraints
    regexs["go_build_constraints"] = re.compile(
        r"^(// \+build.*\n)+\n", re.MULTILINE)
    # strip #!.* from scripts
    regexs["shebang"] = re.compile(r"^(#!.*\n)\n*", re.MULTILINE)
    # Search for generated files
    regexs["generated"] = re.compile('DO NOT EDIT')
    return regexs


def main():
    regexs = get_regexs()
    refs = get_refs()
    filenames = get_files(refs.keys())

    for filename in filenames:
        if not file_update(filename, refs, regexs):
            print(filename, file=sys.stdout)

    return 0


if __name__ == "__main__":
    
    sys.exit(main())
