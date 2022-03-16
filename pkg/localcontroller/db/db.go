/*
Copyright 2021 The KubeEdge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package db

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/pkg/localcontroller/common/constants"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

// Resource defines resource (e.g., dataset, model, jointinferenceservice) table
type Resource struct {
	gorm.Model
	Name       string `gorm:"unique"`
	TypeMeta   string
	ObjectMeta string
	Spec       string
}

var dbClient *gorm.DB

// SaveResource saves resource info in db
func SaveResource(name string, typeMeta, objectMeta, spec interface{}) error {
	var err error

	r := Resource{}

	typeMetaData, _ := json.Marshal(typeMeta)
	objectMetaData, _ := json.Marshal(objectMeta)
	specData, _ := json.Marshal(spec)

	queryResult := dbClient.Where("name = ?", name).First(&r)

	if queryResult.RowsAffected == 0 {
		newR := &Resource{
			Name:       name,
			TypeMeta:   string(typeMetaData),
			ObjectMeta: string(objectMetaData),
			Spec:       string(specData),
		}
		if err = dbClient.Create(newR).Error; err != nil {
			klog.Errorf("failed to save resource(name=%s): %v", name, err)
			return err
		}
		klog.Infof("saved resource(name=%s)", name)
	} else {
		r.TypeMeta = string(typeMetaData)
		r.ObjectMeta = string(objectMetaData)
		r.Spec = string(specData)
		if err := dbClient.Save(&r).Error; err != nil {
			klog.Errorf("failed to update resource(name=%s): %v", name, err)
			return err
		}
		klog.V(2).Infof("updated resource(name=%s)", name)
	}

	return nil
}

// GetResource gets resource info in db
func GetResource(name string) (*Resource, error) {
	r := Resource{}

	queryResult := dbClient.Where("name = ?", name).First(&r)
	if queryResult.RowsAffected == 0 {
		return nil, fmt.Errorf("resource(name=%s) not in db", name)
	}

	return &r, nil
}

// DeleteResource deletes resource info in db
func DeleteResource(name string) error {
	var err error

	r := Resource{}

	queryResult := dbClient.Where("name = ?", name).First(&r)

	if queryResult.RowsAffected == 0 {
		return nil
	}

	if err = dbClient.Unscoped().Delete(&r).Error; err != nil {
		klog.Errorf("failed to delete resource(name=%s): %v", name, err)
		return err
	}
	klog.Infof("deleted resource(name=%s)", name)

	return nil
}

func init() {
	dbClient = getClient()
}

// getClient gets db client
func getClient() *gorm.DB {
	var prefix string
	var ok bool
	if prefix, ok = os.LookupEnv(constants.RootFSMountDirENV); !ok {
		prefix = "/rootfs"
	}

	dbURL := util.AddPrefixPath(prefix, constants.DataBaseURL)

	if _, err := os.Stat(dbURL); err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(dbURL), os.ModePerm); err != nil {
				klog.Errorf("create fold(url=%s) failed, error: %v", filepath.Dir(dbURL), err)
			}
		}
	}

	db, err := gorm.Open(sqlite.Open(dbURL), &gorm.Config{})
	if err != nil {
		klog.Errorf("try to connect the db failed, error: %v", err)
	}

	_ = db.AutoMigrate(&Resource{})

	return db
}
